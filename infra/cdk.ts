#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import {Duration, Stack} from 'aws-cdk-lib';
import {
    FlowLogDestination,
    FlowLogTrafficType,
    InstanceClass,
    InstanceSize,
    InstanceType,
    IVpc,
    Port,
    SubnetType,
    Vpc
} from "aws-cdk-lib/aws-ec2";
import {
    AmiHardwareType,
    AsgCapacityProvider,
    AwsLogDriver,
    Cluster,
    ContainerImage,
    Ec2TaskDefinition,
    EcsOptimizedImage,
    HealthCheck,
    PropagatedTagSource,
    Protocol
} from "aws-cdk-lib/aws-ecs";
import {AutoScalingGroup, GroupMetrics} from "aws-cdk-lib/aws-autoscaling";
import * as path from "path";
import {ICluster} from "aws-cdk-lib/aws-eks";
import {Construct} from "constructs";
import {RetentionDays} from "aws-cdk-lib/aws-logs";
import {HostedZone} from "aws-cdk-lib/aws-route53";
import {Certificate, CertificateValidation} from "aws-cdk-lib/aws-certificatemanager";
import {ApplicationLoadBalancedEc2Service} from "aws-cdk-lib/aws-ecs-patterns";
import {Repository} from "aws-cdk-lib/aws-ecr";
import {HttpsRedirect} from "aws-cdk-lib/aws-route53-patterns";

const root = path.join(__dirname, `..`);

const app = new cdk.App();
const env = {account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION};

const vpcStack = new Stack(app, 'vpc', {env});
const vpc = new Vpc(vpcStack, 'vpc', {
    flowLogs: {
        FlowLogCloudWatch: {
            trafficType: FlowLogTrafficType.ALL,
            destination: FlowLogDestination.toCloudWatchLogs(),
        },
    },
    natGateways: 1,
    maxAzs: 3,
});

const ecsStack = new Stack(app, 'ecs-c7g-large', {env});
const cluster = new Cluster(ecsStack, 'ecs', {
    vpc,
    clusterName: 'ecs-c7g-large',
    containerInsights: true,
});
cluster.connections.allowFromAnyIpv4(Port.tcp(80));
cluster.connections.allowToAnyIpv4(Port.tcp(80));

const INSTANCE = InstanceType.of(InstanceClass.T4G, InstanceSize.LARGE)

const asg = new AutoScalingGroup(ecsStack, "AutoScalingGroup", {
    autoScalingGroupName: "EcsC7gLargeAutoScalingGroup",
    instanceType: INSTANCE,
    vpc,
    groupMetrics: [GroupMetrics.all()],
    minCapacity: 1,
    maxCapacity: 1,
    machineImage: EcsOptimizedImage.amazonLinux2(
        AmiHardwareType.ARM,
    ),
    associatePublicIpAddress: true,
    vpcSubnets: {
        subnetType: SubnetType.PUBLIC
    },
    allowAllOutbound: true,
});
const asgCapacityProvider = new AsgCapacityProvider(ecsStack, `asgCapacityProvider`, {
    autoScalingGroup: asg,
    // canContainersAccessInstanceRole: true,
    capacityProviderName: `C7gLargeAsgCapacityProvider`,
});
cluster.addAsgCapacityProvider(asgCapacityProvider);
asgCapacityProvider.autoScalingGroup.connections.allowFromAnyIpv4(Port.allTcp());
asgCapacityProvider.autoScalingGroup.connections.allowToAnyIpv4(Port.allTcp());

const domain = process.env.DOMAIN;

if (domain) {
    const uptimeRedirect = new Stack(app, 'task-uptime-redirect', {env});
    new HttpsRedirect(uptimeRedirect, 'Redirect', {
        recordNames: [`uptime.task.${domain}`],
        targetDomain: 'task.cronitorstatus.com',
        zone: HostedZone.fromLookup(uptimeRedirect, 'HostedZone', {
            domainName: domain,
        }),
    });
}

interface InferenceServiceProps {
    name: string;
    vpc: IVpc | Vpc;
    cluster: ICluster | Cluster;
    instanceType?: string;
    timeout?: Duration;
    domain?: string;
    subdomain?: string;
    healthCheck?: HealthCheck;
    env?: { [key: string]: string };
}

class Service extends Construct {
    constructor(scope: Construct, id: string, props: InferenceServiceProps) {
        super(scope, id);
        const service = props.name;
        const service_cased = service.charAt(0).toUpperCase() + service.slice(1);
        const service_name = `${service}-service`;

        // const asset = new DockerImageAsset(this, `Image`, {
        //     directory: path.join(root, service),
        //     buildArgs: {
        //         ARCH: INSTANCE.architecture === InstanceArchitecture.ARM_64 ? "arm64" : "amd64",
        //     }
        // });

        // new CfnOutput(this, `ImageUriEcrCfnOut`, {
        //     value: asset.imageUri,
        //     exportName: `${service_cased}ImageUriECR`,
        //     description: `Image URI from ECR`,
        // });

        // const instanceType = props.instanceType || "c7g.large";
        //
        // const placementConstraints = instanceType
        //     ? [
        //         PlacementConstraint.memberOf(
        //             `attribute:ecs.instance-type == ${instanceType}`
        //         ),
        //     ]
        //     : [];

        const taskDefinition = new Ec2TaskDefinition(this, "TaskDef", {
            family: service_name,
        });

        const revision = require('child_process')
            .execSync('git rev-parse HEAD')
            .toString().trim()

        taskDefinition.addContainer("Container", {
            image: ContainerImage.fromEcrRepository(Repository.fromRepositoryName(this, `${service}-ecr`, `${service}`), revision),
            portMappings: [
                {
                    containerPort: 80,
                    protocol: Protocol.TCP,
                },
            ],
            containerName: service_name,
            environment: props.env,
            memoryReservationMiB: 2048,
            logging: new AwsLogDriver({
                streamPrefix: service_name,
                logRetention: RetentionDays.ONE_WEEK,
            }),
        });

        let certificate = undefined;
        let hostedZone = undefined;
        if (props.domain) {
            hostedZone = HostedZone.fromLookup(this, `HostedZone`, {
                domainName: props.domain,
            });

            certificate = new Certificate(this, "Certificate", {
                domainName: `*.${props.domain}`,
                validation: CertificateValidation.fromDns(hostedZone),
            });
        }

        const loadBalancedEcsService = new ApplicationLoadBalancedEc2Service(
            this,
            `Service`,
            {
                cluster,
                serviceName: service_name,
                loadBalancerName: `${service}-load-balancer`,
                desiredCount: 1,
                certificate,
                enableECSManagedTags: true,
                maxHealthyPercent: 400,
                minHealthyPercent: 100,
                // placementConstraints: placementConstraints,
                propagateTags: PropagatedTagSource.TASK_DEFINITION,
                memoryReservationMiB: 2048,
                publicLoadBalancer: true,
                taskDefinition,
                healthCheckGracePeriod: props.timeout || Duration.seconds(60),
                domainName: domain ? props.subdomain || service : undefined,
                domainZone: domain ? hostedZone : undefined,
                redirectHTTP: !!domain,

            }
        );
        loadBalancedEcsService.listener.connections.allowFromAnyIpv4(Port.allTcp());
        loadBalancedEcsService.listener.connections.allowToAnyIpv4(Port.allTcp());

        loadBalancedEcsService.targetGroup.configureHealthCheck(props.healthCheck ?? {
            path: `/ping`,
        });

        // if (hostedZone && loadBalancedEcsService.loadBalancer && certificate) {
        //     new ARecord(this, "DnsRecord", {
        //         recordName: props.subdomain || service,
        //         zone: hostedZone,
        //         target: RecordTarget.fromAlias(
        //             new LoadBalancerTarget(loadBalancedEcsService.loadBalancer)
        //         ),
        //         ttl: Duration.minutes(1),
        //     });
        // }
    }
}

let stack = new Stack(app, "task-api-service", {
    env,
    description: `Task api service`,
});
new Service(stack, "service", {
    name: "task-api",
    vpc,
    cluster,
    env: {
        AWS_REGION: process.env.AWS_REGION!,
        AWS_DEFAULT_REGION: process.env.AWS_REGION!,
        AWS_ACCESS_KEY_ID: process.env.AWS_ACCESS_KEY_ID!,
        AWS_SECRET_ACCESS_KEY: process.env.AWS_SECRET_ACCESS_KEY!,
        S3_BUCKET: process.env.S3_BUCKET!,
        DYNDB_TABLE: process.env.DYNDB_TABLE!,
    },
    domain: process.env.DOMAIN,
    subdomain: "api.task",
});
