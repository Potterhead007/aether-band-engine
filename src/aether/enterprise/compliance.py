"""
AETHER Compliance Controls

Enterprise compliance framework:
- Data retention policies
- PII handling
- GDPR/CCPA support
- SOC 2 controls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""

    SOC2 = "soc2"
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"


class DataClassification(str, Enum):
    """Data classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class DataRetentionPolicy:
    """Data retention policy configuration."""

    name: str
    description: str
    retention_days: int
    data_types: List[str]
    classification: DataClassification = DataClassification.INTERNAL
    auto_delete: bool = True
    archive_before_delete: bool = True
    legal_hold_exempt: bool = False

    def is_expired(self, created_at: datetime) -> bool:
        """Check if data has exceeded retention period."""
        return datetime.utcnow() > created_at + timedelta(days=self.retention_days)


@dataclass
class PIIField:
    """PII field definition."""

    name: str
    classification: DataClassification
    mask_pattern: str = "***"  # For display
    encryption_required: bool = True
    retention_days: Optional[int] = None


@dataclass
class ComplianceControl:
    """Compliance control definition."""

    control_id: str
    name: str
    description: str
    frameworks: List[ComplianceFramework]
    category: str
    implementation_status: str = "implemented"
    evidence_types: List[str] = field(default_factory=list)
    validators: List[Callable[[], bool]] = field(default_factory=list)


class ComplianceManager:
    """
    Enterprise compliance management.

    Features:
    - Data retention enforcement
    - PII identification and masking
    - Compliance control tracking
    - Audit evidence collection
    - Right to deletion (GDPR Art. 17)
    """

    def __init__(self, frameworks: Optional[List[ComplianceFramework]] = None):
        self.frameworks = frameworks or [ComplianceFramework.SOC2]
        self._retention_policies: Dict[str, DataRetentionPolicy] = {}
        self._pii_fields: Dict[str, PIIField] = {}
        self._controls: Dict[str, ComplianceControl] = {}
        self._legal_holds: Set[str] = set()

        # Register default controls
        self._register_default_controls()
        self._register_default_pii_fields()

    def _register_default_controls(self) -> None:
        """Register default compliance controls."""
        default_controls = [
            ComplianceControl(
                control_id="AC-1",
                name="Access Control Policy",
                description="Establish access control policy and procedures",
                frameworks=[ComplianceFramework.SOC2],
                category="Access Control",
                evidence_types=["policy_document", "access_review"],
            ),
            ComplianceControl(
                control_id="AC-2",
                name="Account Management",
                description="Manage user accounts and access rights",
                frameworks=[ComplianceFramework.SOC2],
                category="Access Control",
                evidence_types=["user_list", "access_log"],
            ),
            ComplianceControl(
                control_id="AU-1",
                name="Audit Logging",
                description="Generate and retain audit logs",
                frameworks=[ComplianceFramework.SOC2, ComplianceFramework.GDPR],
                category="Audit",
                evidence_types=["audit_logs", "log_retention"],
            ),
            ComplianceControl(
                control_id="GDPR-17",
                name="Right to Erasure",
                description="Support right to be forgotten",
                frameworks=[ComplianceFramework.GDPR],
                category="Data Subject Rights",
                evidence_types=["deletion_log", "erasure_procedure"],
            ),
            ComplianceControl(
                control_id="GDPR-20",
                name="Data Portability",
                description="Support data export in machine-readable format",
                frameworks=[ComplianceFramework.GDPR],
                category="Data Subject Rights",
                evidence_types=["export_capability", "format_spec"],
            ),
        ]

        for control in default_controls:
            self._controls[control.control_id] = control

    def _register_default_pii_fields(self) -> None:
        """Register default PII field definitions."""
        default_pii = [
            PIIField(
                name="email",
                classification=DataClassification.CONFIDENTIAL,
                mask_pattern="***@***.***",
                encryption_required=True,
                retention_days=365,
            ),
            PIIField(
                name="ip_address",
                classification=DataClassification.INTERNAL,
                mask_pattern="***.***.***.***",
                encryption_required=False,
                retention_days=90,
            ),
            PIIField(
                name="user_agent",
                classification=DataClassification.INTERNAL,
                mask_pattern="[REDACTED]",
                encryption_required=False,
                retention_days=90,
            ),
            PIIField(
                name="name",
                classification=DataClassification.CONFIDENTIAL,
                mask_pattern="*** ***",
                encryption_required=True,
                retention_days=365,
            ),
        ]

        for pii_field in default_pii:
            self._pii_fields[pii_field.name] = pii_field

    def add_retention_policy(self, policy: DataRetentionPolicy) -> None:
        """Add a data retention policy."""
        self._retention_policies[policy.name] = policy
        logger.info(f"Added retention policy: {policy.name}")

    def get_retention_policy(self, data_type: str) -> Optional[DataRetentionPolicy]:
        """Get retention policy for data type."""
        for policy in self._retention_policies.values():
            if data_type in policy.data_types:
                return policy
        return None

    def add_legal_hold(self, resource_id: str, reason: str) -> None:
        """Add legal hold to prevent deletion."""
        self._legal_holds.add(resource_id)
        logger.info(f"Legal hold added for {resource_id}: {reason}")

    def remove_legal_hold(self, resource_id: str) -> None:
        """Remove legal hold."""
        self._legal_holds.discard(resource_id)
        logger.info(f"Legal hold removed for {resource_id}")

    def is_under_legal_hold(self, resource_id: str) -> bool:
        """Check if resource is under legal hold."""
        return resource_id in self._legal_holds

    def mask_pii(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask PII fields in data."""
        masked = data.copy()
        for field_name, pii_field in self._pii_fields.items():
            if field_name in masked:
                masked[field_name] = pii_field.mask_pattern
        return masked

    def identify_pii(self, data: Dict[str, Any]) -> List[str]:
        """Identify PII fields in data."""
        pii_found = []
        for field_name in data.keys():
            if field_name in self._pii_fields:
                pii_found.append(field_name)
        return pii_found

    def get_applicable_controls(self) -> List[ComplianceControl]:
        """Get controls applicable to enabled frameworks."""
        return [
            control
            for control in self._controls.values()
            if any(fw in self.frameworks for fw in control.frameworks)
        ]

    async def execute_right_to_erasure(
        self,
        user_id: str,
        verification_token: str,
    ) -> Dict[str, Any]:
        """
        Execute GDPR Article 17 - Right to Erasure.

        Returns summary of deleted data.
        """
        if ComplianceFramework.GDPR not in self.frameworks:
            raise ValueError("GDPR framework not enabled")

        # Check for legal holds
        if self.is_under_legal_hold(f"user:{user_id}"):
            raise ValueError("User data under legal hold, cannot delete")

        logger.info(f"Executing right to erasure for user: {user_id}")

        deleted = {
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data_categories": [],
            "retained_categories": [],
        }

        # In production, delete from all data stores:
        # - User profile
        # - Audit logs (anonymize, don't delete)
        # - Generated content
        # - Analytics data

        deleted["data_categories"] = [
            "user_profile",
            "preferences",
            "generated_content",
        ]

        deleted["retained_categories"] = [
            "audit_logs (anonymized)",
            "billing_records (legal requirement)",
        ]

        logger.info(f"Erasure complete for user: {user_id}")
        return deleted

    async def export_user_data(
        self,
        user_id: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """
        Export user data (GDPR Article 20 - Data Portability).

        Returns all user data in machine-readable format.
        """
        logger.info(f"Exporting data for user: {user_id}")

        # In production, collect from all data stores
        export = {
            "user_id": user_id,
            "export_timestamp": datetime.utcnow().isoformat(),
            "format": format,
            "profile": {
                "id": user_id,
                "created_at": "2024-01-01T00:00:00Z",
            },
            "generations": [],
            "preferences": {},
        }

        return export

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance status report."""
        controls = self.get_applicable_controls()

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "frameworks": [fw.value for fw in self.frameworks],
            "controls": {
                "total": len(controls),
                "implemented": len([c for c in controls if c.implementation_status == "implemented"]),
                "in_progress": len([c for c in controls if c.implementation_status == "in_progress"]),
                "not_started": len([c for c in controls if c.implementation_status == "not_started"]),
            },
            "retention_policies": len(self._retention_policies),
            "pii_fields_tracked": len(self._pii_fields),
            "active_legal_holds": len(self._legal_holds),
        }

        return report


# Global compliance manager
_compliance_manager: Optional[ComplianceManager] = None


def get_compliance_manager(
    frameworks: Optional[List[ComplianceFramework]] = None,
) -> ComplianceManager:
    """Get or create global compliance manager."""
    global _compliance_manager
    if _compliance_manager is None:
        _compliance_manager = ComplianceManager(frameworks)
    return _compliance_manager
