"""
Anomalies Module - Custom Anomaly Detection Rules

This module allows users to create modular, custom anomaly detection rules.
Each rule is defined as a separate class that inherits from BaseAnomalyRule.
"""
from .base_rule import BaseAnomalyRule, RuleCondition, RuleOperator
from .rule_loader import AnomalyRuleLoader

__all__ = ['BaseAnomalyRule', 'RuleCondition', 'RuleOperator', 'AnomalyRuleLoader']
