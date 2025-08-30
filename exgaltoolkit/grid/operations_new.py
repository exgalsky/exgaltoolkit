"""
Grid operations for distributed computing and FFT operations.

This module now delegates to the modern JAX-native implementation.
"""
from .modern_operations import ModernGridOperations

# Alias for backward compatibility - use the modern implementation
GridOperations = ModernGridOperations
