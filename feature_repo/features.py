from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32
from feast.data_format import ParquetFormat
from datetime import timedelta
import os

# Path to processed parquet file
processed_parquet = os.path.join("..", "data", "processed", "stock_data.parquet")

# Define FileSource
stock_source = FileSource(
    name="stock_source",
    path=processed_parquet,
    timestamp_field="timestamp",
    file_format=ParquetFormat(),
)

# Define Entity
stock = Entity(
    name="stock_name",
    description="Unique stock identifier"
)

# Define FeatureView (schema-based)
stock_fv = FeatureView(
    name="stock_minute_features",
    entities=[stock],
    ttl=timedelta(days=365),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
    ],
    online=True,
    source=stock_source,
)
