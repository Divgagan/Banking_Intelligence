"""
src/database/schema.py
SQLAlchemy ORM models defining all tables.
Run `scripts/create_db.py` to materialise these in PostgreSQL.

Design decisions:
- Raw tables mirror CSV files exactly (no transformations).
- Derived / analytical tables are clearly separated.
- All tables store provenance: created_at, updated_at, source_version.
- Foreign keys enforce referential integrity.
- Indexes on high-cardinality join/filter columns.
"""
from datetime import datetime
from sqlalchemy import (
    BigInteger, Boolean, Column, Date, DateTime, Float,
    ForeignKey, Index, Integer, Numeric, String, Text,
    UniqueConstraint, func,
)
from src.database.engine import Base


# ─────────────────────────────────────────────
# RAW TABLES (mirrors of CSV, read-only)
# ─────────────────────────────────────────────

class CustomerRaw(Base):
    """Mirror of customer_data.csv — never modified after load."""
    __tablename__ = "customers_raw"

    id                          = Column(Integer, primary_key=True, autoincrement=True)
    customer_id                 = Column(BigInteger, nullable=False, index=True)
    age                         = Column(Integer)
    gender                      = Column(String(16))
    location                    = Column(String(128))
    income_bracket              = Column(String(32))
    occupation                  = Column(String(64))
    education_level             = Column(String(32))
    marital_status              = Column(String(32))
    household_size              = Column(Integer)
    acquisition_channel         = Column(String(32))
    customer_segment            = Column(String(32))        # pre-assigned in source
    savings_account             = Column(Boolean)
    credit_card                 = Column(Boolean)
    personal_loan               = Column(Boolean)
    investment_account          = Column(Boolean)
    insurance_product           = Column(Boolean)
    active_products             = Column(Integer)
    app_logins_frequency        = Column(Integer)
    feature_usage_diversity     = Column(Integer)
    bill_payment_user           = Column(Boolean)
    auto_savings_enabled        = Column(Boolean)
    credit_utilization_ratio    = Column(Float)             # nullable — missing for non-CC holders
    international_transactions  = Column(Integer)
    failed_transactions         = Column(Integer)
    tx_count                    = Column(Integer)           # pre-computed aggregate
    avg_tx_value                = Column(Float)
    total_tx_volume             = Column(Numeric(20, 2))
    first_tx                    = Column(Date)
    last_tx                     = Column(Date)
    base_satisfaction           = Column(Float)
    tx_satisfaction             = Column(Float)
    product_satisfaction        = Column(Float)
    satisfaction_score          = Column(Integer)
    nps_score                   = Column(Integer)
    last_survey_date            = Column(Date)
    support_tickets_count       = Column(Integer)
    resolved_tickets_ratio      = Column(Float)
    app_store_rating            = Column(Float)
    feedback_sentiment          = Column(String(16))
    feature_requests            = Column(Text)              # nullable; multi-value text
    complaint_topics            = Column(Text)              # nullable
    clv_segment                 = Column(String(16))        # pre-assigned
    monthly_transaction_count   = Column(Float)             # pre-computed
    average_transaction_value   = Column(Float)             # pre-computed (duplicate of avg_tx_value)
    total_transaction_volume    = Column(Numeric(20, 2))    # pre-computed
    transaction_frequency       = Column(Float)
    last_transaction_date       = Column(Date)
    preferred_transaction_type  = Column(String(32))
    first_transaction_date      = Column(Date)
    weekend_transaction_ratio   = Column(Float)
    avg_daily_transactions      = Column(Float)
    customer_tenure             = Column(Float)             # months
    churn_probability           = Column(Float)             # pre-computed score [0,1] — NOT a label
    customer_lifetime_value     = Column(Float)             # pre-computed

    loaded_at = Column(DateTime, server_default=func.now())


class TransactionRaw(Base):
    """Mirror of transactions_data.csv — never modified after load."""
    __tablename__ = "transactions_raw"

    id          = Column(BigInteger, primary_key=True, autoincrement=True)
    customer_id = Column(BigInteger, nullable=False, index=True)
    date        = Column(Date, nullable=False, index=True)
    amount      = Column(Numeric(20, 2), nullable=False)
    type        = Column(String(32), nullable=False)

    loaded_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_tx_raw_customer_date", "customer_id", "date"),
    )


# ─────────────────────────────────────────────
# CLEANED TABLES
# ─────────────────────────────────────────────

class Customer(Base):
    """
    Cleaned, validated customer records.
    Source: customers_raw with nulls handled, types fixed, duplicates removed.
    """
    __tablename__ = "customers"

    customer_id                 = Column(BigInteger, primary_key=True)
    age_group                   = Column(String(16))        # Derived from age bins
    gender                      = Column(String(16))
    location                    = Column(String(128))
    income_bracket              = Column(String(32))
    occupation                  = Column(String(64))
    education_level             = Column(String(32))
    marital_status              = Column(String(32))
    household_size              = Column(Integer)
    acquisition_channel         = Column(String(32))
    # Product ownership
    savings_account             = Column(Boolean, default=False)
    credit_card                 = Column(Boolean, default=False)
    personal_loan               = Column(Boolean, default=False)
    investment_account          = Column(Boolean, default=False)
    insurance_product           = Column(Boolean, default=False)
    active_products             = Column(Integer)
    bill_payment_user           = Column(Boolean, default=False)
    auto_savings_enabled        = Column(Boolean, default=False)
    # Engagement
    app_logins_frequency        = Column(Integer)
    feature_usage_diversity     = Column(Integer)
    credit_utilization_ratio    = Column(Float)             # imputed where null
    international_transactions  = Column(Integer)
    failed_transactions         = Column(Integer)
    # Satisfaction
    base_satisfaction           = Column(Float)
    tx_satisfaction             = Column(Float)
    product_satisfaction        = Column(Float)
    satisfaction_score          = Column(Integer)
    nps_score                   = Column(Integer)
    support_tickets_count       = Column(Integer)
    resolved_tickets_ratio      = Column(Float)
    app_store_rating            = Column(Float)
    feedback_sentiment          = Column(String(16))
    has_complaints              = Column(Boolean)           # derived from complaint_topics
    has_feature_requests        = Column(Boolean)           # derived from feature_requests
    # Tenure
    customer_tenure_months      = Column(Float)
    preferred_transaction_type  = Column(String(32))
    # Source pre-computed scores (for reference only — not used as ML targets)
    source_churn_probability    = Column(Float)
    source_clv                  = Column(Float)
    source_segment              = Column(String(32))
    source_clv_segment          = Column(String(16))

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Transaction(Base):
    """
    Cleaned transaction records.
    Duplicates removed. date parsed. customer_id validated against customers.
    """
    __tablename__ = "transactions"

    id          = Column(BigInteger, primary_key=True, autoincrement=True)
    customer_id = Column(BigInteger, ForeignKey("customers.customer_id"), nullable=False)
    date        = Column(Date, nullable=False)
    amount      = Column(Numeric(20, 2), nullable=False)
    type        = Column(String(32), nullable=False)

    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_tx_customer_date", "customer_id", "date"),
        Index("ix_tx_date", "date"),
    )


# ─────────────────────────────────────────────
# ANALYTICAL / DERIVED TABLES
# ─────────────────────────────────────────────

class CustomerFeatures(Base):
    """
    Engineered features computed from raw transactions + customer attributes.
    One row per customer. Rebuilt whenever the feature pipeline runs.
    """
    __tablename__ = "customer_features"

    customer_id                 = Column(BigInteger, ForeignKey("customers.customer_id"), primary_key=True)
    feature_version             = Column(String(16), default="v1.0")

    # --- RFM-style features ---
    recency_days                = Column(Integer)           # days since last transaction
    frequency_30d               = Column(Integer)           # tx count in last 30 days
    frequency_90d               = Column(Integer)
    frequency_total             = Column(Integer)
    monetary_total              = Column(Numeric(20, 2))    # sum of all amounts
    monetary_avg                = Column(Float)
    monetary_std                = Column(Float)

    # --- Transaction type breakdown ---
    pct_deposit                 = Column(Float)
    pct_withdrawal              = Column(Float)
    pct_transfer                = Column(Float)
    pct_payment                 = Column(Float)

    # --- Temporal patterns ---
    weekend_ratio               = Column(Float)
    avg_tx_per_month            = Column(Float)
    active_months               = Column(Integer)           # months with at least 1 tx

    # --- Volatility ---
    amount_cv                   = Column(Float)             # coefficient of variation of amounts

    # --- Engagement composite ---
    engagement_score            = Column(Float)             # normalised composite

    computed_at = Column(DateTime, server_default=func.now())


class CustomerSegment(Base):
    """Segment assignment output from the clustering pipeline."""
    __tablename__ = "customer_segments"

    customer_id         = Column(BigInteger, ForeignKey("customers.customer_id"), primary_key=True)
    segment_id          = Column(Integer, nullable=False)
    segment_label       = Column(String(64))                # human-readable name assigned post-analysis
    model_version       = Column(String(16), default="kmeans_v1")
    silhouette_score    = Column(Float)
    assigned_at         = Column(DateTime, server_default=func.now())


class ModelPrediction(Base):
    """Versioned scores or indicators with explicit prediction_type metadata."""
    __tablename__ = "model_predictions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    customer_id     = Column(BigInteger, ForeignKey("customers.customer_id"), nullable=False, index=True)
    model_name      = Column(String(64), nullable=False)
    model_version   = Column(String(16), nullable=False)
    prediction_type = Column(String(32))                    # churn_risk | propensity | etc.
    score           = Column(Float)                         # interpretation depends on prediction_type
    label           = Column(Integer)                       # 0 / 1 (if thresholded)
    threshold_used  = Column(Float)
    predicted_at    = Column(DateTime, server_default=func.now())
    feature_version = Column(String(16))

    __table_args__ = (
        Index("ix_pred_customer_model", "customer_id", "model_name"),
    )


class ProductPropensity(Base):
    """Per-customer product scores, with score_type recording their meaning."""
    __tablename__ = "product_propensity"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    customer_id     = Column(BigInteger, ForeignKey("customers.customer_id"), nullable=False, index=True)
    product_name    = Column(String(64), nullable=False)    # savings_account | credit_card | etc.
    score           = Column(Float)                         # 0–1
    score_type      = Column(String(32))                    # model | rule_based | heuristic
    model_version   = Column(String(16))
    computed_at     = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("customer_id", "product_name", name="uq_propensity_customer_product"),
    )


class RecommendedAction(Base):
    """Next-best-action engine output — one active recommendation per customer."""
    __tablename__ = "recommended_actions"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    customer_id         = Column(BigInteger, ForeignKey("customers.customer_id"), nullable=False, index=True)
    action_code         = Column(String(64), nullable=False)
    action_description  = Column(Text)
    reason              = Column(Text)
    evidence            = Column(Text)                      # JSON string of evidence list
    priority_score      = Column(Float)
    requires_human_review = Column(Boolean, default=True)
    engine_version      = Column(String(16), default="rule_v1")
    recommended_at      = Column(DateTime, server_default=func.now())
    expires_at          = Column(DateTime)
    outcome             = Column(String(32))                # accepted | rejected | pending | expired


class ModelRegistry(Base):
    """Tracks all trained model artefacts."""
    __tablename__ = "model_registry"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    model_name      = Column(String(64), nullable=False)
    model_version   = Column(String(16), nullable=False)
    model_type      = Column(String(32))                    # classification | clustering | etc.
    artifact_path   = Column(String(256))
    metrics         = Column(Text)                          # JSON
    feature_version = Column(String(16))
    trained_at      = Column(DateTime, server_default=func.now())
    is_active       = Column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("model_name", "model_version", name="uq_model_version"),
    )


class AuditLog(Base):
    """Immutable event log for all system actions."""
    __tablename__ = "audit_logs"

    id          = Column(BigInteger, primary_key=True, autoincrement=True)
    event_type  = Column(String(64), nullable=False, index=True)
    entity_type = Column(String(32))
    entity_id   = Column(String(64))
    description = Column(Text)
    event_metadata = Column(Text)                           # JSON
    created_at  = Column(DateTime, server_default=func.now(), index=True)
