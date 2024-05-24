import torch
import importlib
from dataclasses import dataclass
from typing import List, Type, Any, Callable, Optional, Union
from loguru import logger
from utils.decorators import *


@dataclass(slots=True)
class BaseFactory:
    config: dict

    def load_class(self, category: str, lib_check: Type[Any], name: str) -> Type[Any]:
        """Loading Classes from a Module"""
        try:
            return getattr(lib_check, name) if hasattr(lib_check, name) else getattr(importlib.import_module(f"utils.implements.{category}"), name)
        except ImportError as e:
            logger.error(f"Error importing module {category}: {e}")
            raise
        except AttributeError as e:
            logger.error(f"Class {name} not found in module {category}: {e}")
            raise

    def create_instance(self, cls: Type[Any], name: str, target: Any = None) -> Any:
        """Create Instance"""
        arg_dict = self.config.get(name, {})
        logger.info(f'Creating {cls.__name__} instance with args: {arg_dict}')
        return cls(target, **arg_dict) if target else cls(**arg_dict)

    def get_instances(self, category: str, lib_check: Type[Any], instance_creator: Callable) -> List[Any]:
        """Create Instance List"""
        return [instance_creator(self.load_class(category, lib_check, name), name) for name in self.config["name"]]


@logger_wraps()
@dataclass(slots=True)
class CriterionFactory(BaseFactory):
    device: torch.device
    
    def get_criterions(self) -> List[Type[Any]]:
        return self.get_instances("criterions", torch.nn, lambda cls, name: self.create_instance(cls, name, self.device))


@logger_wraps()
@dataclass(slots=True)
class OptimizerFactory(BaseFactory):
    parameters_policy: Any

    def get_optimizers(self) -> List[Type[Any]]:
        return self.get_instances("optimizers", torch.optim, lambda cls, name: self.create_instance(cls, name, self.parameters_policy))


@logger_wraps()
@dataclass(slots=True)
class SchedulerFactory(BaseFactory):
    optimizers: List[Any]

    def get_schedulers(self) -> List[Type[Any]]:
        optimizers = self.optimizers
        if len(self.optimizers) == 1 and len(self.config["name"]) > 1:
            optimizers = self.optimizers * len(self.config["name"]) 

        return [self.create_instance(self.load_class("schedulers", torch.optim.lr_scheduler, name), name, optimizer) 
                for name, optimizer in zip(self.config["name"], optimizers)]