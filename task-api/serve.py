import json
import logging
import os
import time
from decimal import Decimal
from enum import Enum
from random import shuffle
from typing import List

import boto3
import shortuuid
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rich import traceback
from rich.console import Console
from rich.logging import RichHandler

# To avoid double logging with uvicorn - known issue in GitHub.
from starlette.responses import RedirectResponse

_uvilog = logging.getLogger("uvicorn.error")
_uvilog.propagate = False

# Setup the logger and handler.
traceback.install(show_locals=True)
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("rich")

console = Console()


class Serializable(BaseModel):
    class Config:
        allow_mutation = False
        extra = 'forbid'
        use_enum_values = True
        validate_assignment = True


class ErrorType(str, Enum):
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    OVERLAP = "OVERLAP"


class Error(Serializable):
    type: ErrorType = Field(...,
                            title="Error type",
                            description="Error type",
                            example="OUT_OF_BOUNDS")
    index: List[int] = Field(...,
                             title="Index",
                             description="Index",
                             example=[0, 1])


class Bbox(Serializable):
    x1: Decimal = Field(...,
                        title="Coordinate X",
                        description="The x coordinate of the top left corner of the bounding box.",
                        example=0.3,
                        ge=0,
                        le=1)
    y1: Decimal = Field(...,
                        title="Coordinate Y",
                        description="The y coordinate of the top left corner of the bounding box.",
                        example=0.3,
                        ge=0,
                        le=1)
    x_off: Decimal = Field(...,
                           title="Width",
                           description="The width of the bounding box.",
                           example=0.3,
                           ge=0,
                           le=1)
    y_off: Decimal = Field(...,
                           title="Height",
                           description="The height of the bounding box.",
                           example=0.3,
                           ge=0,
                           le=1)

    class Config:
        title = "Bounding Box"

    def is_out_of_bounds(self) -> bool:
        return self.x1 < 0 or self.x1 + self.x_off > 1 or self.y1 < 0 or self.y1 + self.y_off > 1

    def is_overlapping(self, other) -> bool:
        return not (
                self.x1 > other.x1 + other.x_off or self.x1 + self.x_off < other.x1 or self.y1 > other.y1 + other.y_off or self.y1 + self.y_off < other.y1)


class Item(Serializable):
    boxes: List[Bbox] = Field(...,
                              title="List of bounding boxes",
                              description="List of bounding boxes",
                              example=[Bbox(x1=0.3, y1=0.3, x_off=0.3, y_off=0.3)])
    errors: List[Error] = Field(...,
                                title="Errors",
                                description="Errors such as out of bounds or overlapping",
                                example=[Error(type=ErrorType.OUT_OF_BOUNDS, index=[0])])
    timestamp: int = Field(...,
                           title="Timestamp",
                           description="Timestamp",
                           example=int(time.time()))


class StatefulItem(Item):
    id: str = Field(...,
                    title="Identifier",
                    description="Unique S3 URI",
                    example="s3://bucket/uuid")

    class Config:
        title = "Stateful Item"


class Heartbeat(Serializable):
    status: str = Field(...,
                        title="Status",
                        description="Status",
                        example="OK")


def load_data(file: str) -> List[Bbox]:
    with open(file) as f:
        data = json.load(f)

    return list(map(lambda d: Bbox(x1=d["x1"], y1=d["y1"], x_off=d["x_off"], y_off=d["y_off"]), data))


def validate(data: List[Bbox]) -> List[Error]:
    ret: List[Error] = []
    for i, bbox in enumerate(data):
        if bbox.is_out_of_bounds():
            ret.append(Error(type=ErrorType.OUT_OF_BOUNDS, index=[i]))
        for j, other in enumerate(data):
            if i == j:
                continue
            if bbox.is_overlapping(other):
                if any(ErrorType.OVERLAP == e.type and i in e.index and j in e.index for e in ret):
                    continue
                ret.append(Error(type=ErrorType.OVERLAP, index=[i, j]))
    return ret


def get_table(dyndb, table_name):
    try:
        return dyndb.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
            ProvisionedThroughput={"ReadCapacityUnits": 1, "WriteCapacityUnits": 1},
        )
    except dyndb_client.exceptions.ResourceInUseException:
        return dyndb.Table(table_name)


def get_data(file: str) -> List[Bbox]:
    data = load_data(file)
    shuffle(data)
    return data[:3]


def get_env(key, default=None):
    env = os.environ.get(key, default)
    if env is None:
        raise Exception(f"{key} is not set")
    return env


# AWS
s3 = boto3.resource("s3")
bucket = s3.create_bucket(Bucket=get_env("S3_BUCKET"))

# Only DynamoDB client can access exceptions
dyndb_client = boto3.client("dynamodb")
dyndb = boto3.resource("dynamodb")
table = get_table(dyndb, get_env("DYNDB_TABLE"))

# FAST API
api = FastAPI(title="TASK API",
              description="""
                API for the task.
                
                Works standalone and integrates with the frontend as well.
                """,
              version="0.1.0",
              terms_of_service="https://github.com/israelg99",
              license_info={
                  "name": "MIT License",
                  "url": "https://github.com/israelg99",
              },
              contact={
                  "name": "Julian Gilyadov",
                  "url": "https://github.com/israelg99",
                  "email": "israelg99@gmail.com",
              },
              openapi_tags=[
                  {
                      "name": "Health",
                      "description": "Health of the service",
                  },
                  {
                      "name": "CRUD",
                      "description": "Stateful CRUD operations",
                  },
                  {
                      "name": "Stateless",
                      "description": "Stateless operations",
                  },
                  {
                      "name": "Documentation",
                      "description": "Documentation of the service",
                  },
              ],
              servers=[
                  {
                      "url": "http://localhost:8000/",
                      "description": "Localhost",
                  },
                  {
                      "url": "http://task-api-load-balancer-1164277279.us-east-1.elb.amazonaws.com/",
                      "description": "Production",
                  },
              ],
              docs_url="/swagger",
              redoc_url="/redoc",
              )

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.put("/",
         response_model=StatefulItem,
         tags=["CRUD"],
         summary="Ingests and processes data",
         description="Ingests and processes data")
async def put(file: UploadFile = File(..., title="File", description="File to process", example="file.jpg")):
    data = get_data("sample-data.json")

    errors = validate(data)

    key = shortuuid.uuid()
    bucket.put_object(Key=key, Body=await file.read())
    s3_uri = f"s3://{bucket.name}/{key}"

    log.info(f"Uploaded {s3_uri}")

    item = StatefulItem(
        id=s3_uri,
        boxes=data,
        timestamp=int(time.time()),
        errors=errors)

    log.info(f"Storing {item}")

    table.put_item(Item=item.dict())

    return item


@api.post("/",
          response_model=Item,
          tags=["Stateless"],
          summary="Processes data",
          description="Processes data")
async def post(data: List[Bbox] = Body(..., title="Data", description="Data to process",
                                       example=[{"x1": 0.1, "y1": 0.1, "x_off": 0.3, "y_off": 0.3}])):
    errors = validate(data)

    item = Item(
        boxes=data,
        timestamp=int(time.time()),
        errors=errors)

    return item


@api.get("/ping",
         response_model=Heartbeat,
         tags=["Health"],
         summary="Heartbeat check",
         description="Heartbeat check")
async def ping():
    return {"status": "ok"}


@api.get("/docs",
         tags=["Documentation"],
         summary="API documentation",
         description="API documentation",
         response_class=RedirectResponse)
async def docs():
    return "https://task-api.readme.io/reference/"


@api.get("/",
         tags=["Documentation"],
         summary="API documentation",
         description="API documentation",
         response_class=RedirectResponse)
async def root():
    return "https://task-api.readme.io/reference/"
