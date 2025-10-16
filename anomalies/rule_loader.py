"""
Rule Loader - Dynamically loads and manages custom anomaly detection rules

This module discovers and loads all custom rules from the anomalies directory
"""
import os
import importlib
import inspect
from typing import List, Dict, Type
from .base_rule import BaseAnomalyRule


class AnomalyRuleLoader:
    """
    Dynamically loads custom anomaly rules from the anomalies directory

    Usage:
        loader = AnomalyRuleLoader()
        rules = loader.load_all_rules()
        enabled_rules = loader.get_enabled_rules()
    """

    def __init__(self, rules_directory: str = None):
        """
        Initialize the rule loader

        Args:
            rules_directory: Directory containing rule files (defaults to anomalies/)
        """
        if rules_directory is None:
            rules_directory = os.path.dirname(os.path.abspath(__file__))

        self.rules_directory = rules_directory
        self.loaded_rules: List[BaseAnomalyRule] = []
        self.rule_registry: Dict[str, Type[BaseAnomalyRule]] = {}

    def discover_rule_modules(self) -> List[str]:
        """
        Discover all Python files in the rules directory

        Returns:
            List of module names (without .py extension)
        """
        modules = []

        for filename in os.listdir(self.rules_directory):
            if filename.endswith('.py') and not filename.startswith('_'):
                if filename not in ['base_rule.py', 'rule_loader.py']:
                    module_name = filename[:-3]
                    modules.append(module_name)

        return modules

    def load_rule_from_module(self, module_name: str) -> List[BaseAnomalyRule]:
        """
        Load all rule classes from a module

        Args:
            module_name: Name of the module to load

        Returns:
            List of instantiated rule objects
        """
        rules = []

        try:
            # Import the module
            module = importlib.import_module(f'anomalies.{module_name}')

            # Find all classes that inherit from BaseAnomalyRule
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseAnomalyRule) and
                    obj is not BaseAnomalyRule and
                    not inspect.isabstract(obj)):

                    # Instantiate the rule
                    try:
                        rule_instance = obj()
                        rules.append(rule_instance)
                        self.rule_registry[rule_instance.name] = obj
                        print(f"✓ Loaded rule: {rule_instance.name}")
                    except Exception as e:
                        print(f"✗ Failed to instantiate {name}: {e}")

        except Exception as e:
            print(f"✗ Failed to load module {module_name}: {e}")

        return rules

    def load_all_rules(self) -> List[BaseAnomalyRule]:
        """
        Load all anomaly rules from the directory

        Returns:
            List of all loaded rule instances
        """
        self.loaded_rules.clear()
        self.rule_registry.clear()

        print(f"\n🔍 Discovering anomaly rules in: {self.rules_directory}")

        modules = self.discover_rule_modules()
        print(f"📦 Found {len(modules)} rule modules: {modules}")

        for module_name in modules:
            rules = self.load_rule_from_module(module_name)
            self.loaded_rules.extend(rules)

        print(f"✅ Successfully loaded {len(self.loaded_rules)} rules\n")

        return self.loaded_rules

    def get_enabled_rules(self) -> List[BaseAnomalyRule]:
        """Get only enabled rules"""
        return [rule for rule in self.loaded_rules if rule.enabled]

    def get_rule_by_name(self, name: str) -> BaseAnomalyRule:
        """Get a specific rule by name"""
        for rule in self.loaded_rules:
            if rule.name == name:
                return rule
        return None

    def enable_rule(self, name: str):
        """Enable a rule by name"""
        rule = self.get_rule_by_name(name)
        if rule:
            rule.enabled = True

    def disable_rule(self, name: str):
        """Disable a rule by name"""
        rule = self.get_rule_by_name(name)
        if rule:
            rule.enabled = False

    def get_rules_summary(self) -> List[Dict]:
        """Get summary of all loaded rules"""
        return [
            {
                'name': rule.name,
                'description': rule.description,
                'severity': rule.severity,
                'enabled': rule.enabled,
                'parameters': rule.parameters
            }
            for rule in self.loaded_rules
        ]

    def reload_rules(self):
        """Reload all rules (useful for development)"""
        importlib.invalidate_caches()
        return self.load_all_rules()
