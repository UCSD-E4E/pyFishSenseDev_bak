from abc import ABC, abstractmethod, abstractproperty
from argparse import ArgumentParser, Namespace

from argument_parser_builder import ArgumentParserBuilder, ParsedArguments

# Using ABC creates an abstract class


class CliCommand(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._parser: ArgumentParser = None
        self._builder: ArgumentParserBuilder = None

    @abstractproperty
    def name(self) -> str:
        raise NotImplementedError()

    @abstractproperty
    def help(self) -> str:
        raise NotImplementedError()

    @property
    def parser(self):
        return self._parser

    @property
    def builder(self):
        return self._builder

    @parser.setter
    def parser(self, parser: ArgumentParser):
        self._parser = parser
        self._builder = ArgumentParserBuilder(parser)
        self._set_parser(parser, self._builder)

    @abstractmethod
    def execute(
        self, args: Namespace, parsed_arguments: ParsedArguments
    ) -> int or None:
        raise NotImplementedError()

    def _set_parser(self, parser: ArgumentParser, builder: ArgumentParserBuilder):
        pass
