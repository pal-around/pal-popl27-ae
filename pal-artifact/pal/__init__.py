"""PAL (Probabilistic Array Language).

The package re-exports the public surface of :mod:`pal.src.main` so user
programs and examples can write ``from pal import pal, Bool, Fin, ...`` directly.
"""

from .src.main import (
    MAIN_KEYS,
    Bool,
    Fin,
    Fin2,
    Fin3,
    Fin4,
    Fin5,
    Fin6,
    Fin7,
    Fin8,
    Fin9,
    _pal,
    _pal_impl,
    add_arguments,
    argument_parser,
    arguments,
    condition,
    init,
    markov,
    observe,
    pal,
    parse_command_line_arguments,
    sample,
    time_execution,
)

__all__ = [
    "Bool",
    "Fin",
    "Fin2",
    "Fin3",
    "Fin4",
    "Fin5",
    "Fin6",
    "Fin7",
    "Fin8",
    "Fin9",
    "MAIN_KEYS",
    "add_arguments",
    "argument_parser",
    "arguments",
    "condition",
    "init",
    "markov",
    "observe",
    "pal",
    "parse_command_line_arguments",
    "sample",
    "time_execution",
]
