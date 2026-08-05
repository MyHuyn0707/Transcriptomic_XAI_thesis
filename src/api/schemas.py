"""
src/api/schemas.py

Pydantic response models mirroring the shapes the React frontend expects
(see frontend/src/App.tsx and components/*.tsx). Kept permissive (extra
fields allowed via dict passthrough) since the underlying JSON files already
carry the right keys — these models exist to document the contract in one
place, not to re-validate every field strictly.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class DatasetInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    platform: str = ""
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    n_classes: Optional[int] = None
    class_labels: List[str] = []
    description: str = ""
    fs_models: Dict[str, List[str]] = {}


class DatasetOverview(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset: str
    class_labels: List[str]
    train_class_counts: Dict[str, int]
    test_class_counts: Dict[str, int]
    n_samples_total: int
    n_train: int
    n_test: int
    test_size: float
    min_samples_per_class: int
    visualizations: Dict[str, str] = {}


class ExtractionStats(BaseModel):
    model_config = ConfigDict(extra="allow")

    framework: str
    dataset_name: str
    feature_selection: str
    n_samples: int
    n_original_features: int
    n_selected_features: int
    runtime_seconds: float


class TrainJobResponse(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # "running" | "done" | "error"
    log: List[str]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ModelStats(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name: str
    best_run_test_metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    class_labels: List[str]
    n_rules: int
    hyperparams: Dict[str, Any]
    confusion_matrix_png: Optional[str] = None


class RuleCard(BaseModel):
    model_config = ConfigDict(extra="allow")

    rule_id: int
    model: Optional[str] = None
    consequent: Dict[str, Any]
    antecedent: List[Dict[str, Any]]
    antecedent_raw: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    provenance: Dict[str, Any]
    text: str
    explanation: Optional[str] = None


class GeneCard(BaseModel):
    model_config = ConfigDict(extra="allow")

    gene: str
    gene_title: str = ""
    entrez_id: str = ""
    genbank_acc: str = ""
    refseq: str = ""
    species: str = ""
    go_biological_process: str = ""
    go_cellular_component: str = ""
    go_molecular_function: str = ""
    n_rules: int = 0
    classes: str = ""
    probes: str = ""
    annotated: bool = False
    description_vn: Optional[str] = None


class TestSample(BaseModel):
    sample_id: str
    true_label: str
    row_index: int


class PredictRequest(BaseModel):
    fs_method: str
    model: str
    sample_id: str
    run_id: Optional[str] = None
    split_params: Optional[Dict[str, Any]] = None


class MatchedRule(BaseModel):
    model_config = ConfigDict(extra="allow")

    rule_id: int
    text: str
    matched: bool
    consequent_label: str
    metrics: Dict[str, Any]
    sample_values: Dict[str, float]
    explanation: Optional[str] = None


class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    classification: str
    class_description: Optional[str] = None
    class_display_name: Optional[str] = None
    class_display_names: Dict[str, str] = {}
    rule_prediction: Optional[str] = None
    rule_prediction_description: Optional[str] = None
    rule_prediction_display_name: Optional[str] = None
    matched_count: int
    rules: List[MatchedRule]
    class_votes: Dict[str, Any]
    biomedical_summary: str
    biomedical_rationale: str
    biomedical_model_vs_rule: str
    biomedical_disclaimer: str
    llm_used: bool
    true_label: str
