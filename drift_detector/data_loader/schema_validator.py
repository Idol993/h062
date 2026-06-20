from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, ValidationError


class FeatureSchema(BaseModel):
    name: str
    type: str
    required: bool = True
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class ValidationResult(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)
    extra_columns: List[str] = Field(default_factory=list)
    type_mismatches: List[str] = Field(default_factory=list)
    missing_values: Dict[str, int] = Field(default_factory=dict)


class SchemaValidator:
    def __init__(self, schemas: Optional[List[FeatureSchema]] = None):
        self.schemas = schemas or []
        self.schema_map: Dict[str, FeatureSchema] = (
            {s.name: s for s in self.schemas} if self.schemas else {}
        )

    def add_schema(self, schema: FeatureSchema) -> None:
        self.schemas.append(schema)
        self.schema_map[schema.name] = schema

    def infer_from_baseline(self, baseline_data: pd.DataFrame) -> None:
        self.schemas = []
        self.schema_map = {}

        for column in baseline_data.columns:
            series = baseline_data[column]
            if pd.api.types.is_numeric_dtype(series):
                schema = FeatureSchema(
                    name=column,
                    type="numerical",
                    required=True,
                    min_value=float(series.min()),
                    max_value=float(series.max()),
                )
            else:
                categories = series.dropna().unique().tolist()
                schema = FeatureSchema(
                    name=column,
                    type="categorical",
                    required=True,
                    allowed_values=[str(v) for v in categories],
                )
            self.add_schema(schema)

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        errors = []
        warnings = []
        missing_columns = []
        extra_columns = []
        type_mismatches = []
        missing_values = {}

        expected_columns = [s.name for s in self.schemas]
        actual_columns = list(data.columns)

        for col in expected_columns:
            if col not in actual_columns:
                missing_columns.append(col)
                if self.schema_map[col].required:
                    errors.append(f"Missing required column: {col}")
                else:
                    warnings.append(f"Missing optional column: {col}")

        for col in actual_columns:
            if col not in expected_columns:
                extra_columns.append(col)
                warnings.append(f"Unexpected column: {col}")

        for col in actual_columns:
            if col not in self.schema_map:
                continue

            schema = self.schema_map[col]
            series = data[col]

            null_count = int(series.isnull().sum())
            if null_count > 0:
                missing_values[col] = null_count
                null_ratio = null_count / len(series)
                if null_ratio > 0.5:
                    errors.append(f"High missing values in {col}: {null_ratio:.1%}")
                elif null_ratio > 0.1:
                    warnings.append(f"Moderate missing values in {col}: {null_ratio:.1%}")

            if schema.required and null_count == len(series):
                errors.append(f"All values missing in required column: {col}")

            series_clean = series.dropna()
            if len(series_clean) == 0:
                continue

            if schema.type == "numerical":
                if not pd.api.types.is_numeric_dtype(series_clean):
                    type_mismatches.append(
                        f"Column {col} should be numerical but is {series_clean.dtype}"
                    )
                    errors.append(
                        f"Type mismatch in {col}: expected numerical, got {series_clean.dtype}"
                    )
                else:
                    if schema.min_value is not None:
                        below_min = (series_clean < schema.min_value).sum()
                        if below_min > 0:
                            warnings.append(
                                f"{below_min} values below min ({schema.min_value}) in {col}"
                            )
                    if schema.max_value is not None:
                        above_max = (series_clean > schema.max_value).sum()
                        if above_max > 0:
                            warnings.append(
                                f"{above_max} values above max ({schema.max_value}) in {col}"
                            )
            elif schema.type == "categorical":
                if schema.allowed_values:
                    str_values = series_clean.astype(str)
                    invalid = ~str_values.isin(schema.allowed_values)
                    invalid_count = int(invalid.sum())
                    if invalid_count > 0:
                        invalid_cats = str_values[invalid].unique()[:5].tolist()
                        warnings.append(
                            f"{invalid_count} invalid categories in {col}: {invalid_cats}"
                        )

        valid = len(errors) == 0
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            missing_columns=missing_columns,
            extra_columns=extra_columns,
            type_mismatches=type_mismatches,
            missing_values=missing_values,
        )

    def validate_against_baseline(
        self,
        production_data: pd.DataFrame,
        baseline_data: pd.DataFrame,
    ) -> ValidationResult:
        self.infer_from_baseline(baseline_data)
        return self.validate(production_data)
