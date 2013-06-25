#!/usr/bin/env python
#
# Copyright 2013 Rackspace US, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import operator
import re

# Stupid tokenizer.  Use:
#
# ft.parse(filter)
# ft.scan() returns (token,value) tuple, destroying token
# ft.peek() returns (token,value) tuple, leaving token on stack.
#           if you peek and decide to process the token, then
#           make sure you eat the token with a scan()!
#
# Since the entire filter is pre-tokenized, one could add
# arbitrary lookahead.  I don't need it tho.
#
# it's trivially easy to confuse this lexer.
#
class Tokenizer(object):
    def __init__(self):
        self.tokens = []
        self.remainer = ''
        classname = self.__class__.__name__.lower()
        self.logger = logging.getLogger('%s.%s' % (__name__, classname))

        self.scanner = re.Scanner([
            (r"\+", self.arith_op),
            (r"-", self.arith_op),
            (r"/", self.arith_op),
            (r"\*", self.arith_op),
            (r"==", self.logical_op),
            (r"!=", self.logical_op),
            (r"=", self.arith_op),
            (r"or", self.logical_op),
            (r"and", self.logical_op),
            (r"None", self.none),
            (r"in ", self.logical_op),
            (r"True", self.bool_lvalue),
            (r"False", self.bool_lvalue),
            (r",", self.comma),
            (r"\.", self.dot),
            (r"[ \t\n]+", None),
            (r"[0-9]+", self.number),
            (r"\(", self.open_paren),
            (r"\)", self.close_paren),
            (r"'([^'\\]*(?:\\.[^'\\]*)*)'", self.qstring),
            (r'"([^"\\]*(?:\\.[^"\\]*)*)"', self.qstring),
            (r"\<\=|\>\=", self.logical_op),
            (r"\<|\>", self.logical_op),
            (r"[A-Za-z{][A-Za-z0-9_\-{}]*", self.identifier),
            (r"\[", self.open_bracket),
            (r"\]", self.close_bracket)
        ])

    def parse(self, input_expression):
        self.tokens, self.remainder = self.scanner.scan(input_expression)
        self.tokens.append(('EOF', None))

        if self.remainder != '':
            raise RuntimeError(
                'Cannot parse.  Input: %s\nTokens: %s\nRemainder %s' %
                (input_expression, self.tokens, self.remainder))

        self.logger.debug('Tokenized %s as %s' %
                          (input_expression, self.tokens))
        return True

    def scan(self):
        self.logger.debug('popping token %s' % str(self.peek()))
        return self.tokens.pop(0)

    def peek(self):
        return self.tokens[0]

    # token generators
    def arith_op(self, scanner, token):
        return 'ARITH_OP', token

    def open_bracket(self, scanner, token):
        return 'OPENBRACKET', token

    def close_bracket(self, scanner, token):
        return 'CLOSEBRACKET', token

    def logical_op(self, scanner, token):
        return 'LOGICAL_OP', token.upper()

    def number(self, scanner, token):
        return 'NUMBER', token

    def identifier(self, scanner, token):
        return 'IDENTIFIER', token

    def bool_lvalue(self, scanner, token):
        return 'LVALUE', token.upper()

    def qstring(self, scanner, token):
        whatquote = token[0]
        escaped_quote = '\\"'
        if whatquote == '\'':
            escaped_quote = "\\'"

        return 'LVALUE', token[1:-1].replace(escaped_quote, whatquote)

    def open_paren(self, scanner, token):
        return 'OPENPAREN', token

    def close_paren(self, scanner, token):
        return 'CLOSEPAREN', token

    def or_op(self, scanner, token):
        return 'OR', token

    def and_op(self, scanner, token):
        return 'AND', token

    def comma(self, scanner, token):
        return 'COMMA', token

    def dot(self, scanner, token):
        return 'DOT', token

    def none(self, scanner, token):
        return 'NONE', token


def logwrapper(func):
    def f(self, *args, **kwargs):
        fname = func.__name__

        self.logger.debug('Entering %s: %s' % (fname, self.tokenizer.peek()))
        retval = func(self, *args, **kwargs)
        self.logger.debug('Exiting %s' % (fname, ))
        return retval

    return f


#
# This is a pretty trivial implementation.  The production
# rules are as follows:
#
# phrase -> {typedef}? andexpr EOF
# andexpr -> orexpr { T_AND orexpr }
# orexpr -> expr { T_OR expr }
# expr -> T_OPENPAREN andexpr T_CLOSEPAREN | criterion
# criterion -> evaluable_item [ { uneg } op evaluable_item ]
# evalable_item -> function(evaluable_item, e_i, ...) | identifier | value
#
# field -> datatype.value
# op -> '=', '<', '>'
# value -> number | string | openbracket e_i [, e_i...] closebracket
#

class AstBuilder(object):
    def __init__(self, input_expression=None, symtable=None):
        self.tokenizer = Tokenizer()

        self.input_expression = input_expression
        classname = self.__class__.__name__.lower()
        self.logger = logging.getLogger('%s.%s' % (__name__, classname))
        self.logger.debug('New builder on expression: %s' %
                          self.input_expression)
        self.symtable = symtable

    def set_input(self, input_expression):
        self.logger.debug('resetting input expression to %s' %
                          input_expression)
        self.input_expression = input_expression

    def build(self):
        self.tokenizer.parse(self.input_expression)
        root_node = self.parse()
        return root_node

    def parse(self):
        return self.parse_phrase()

    def eval(self, symtable=None):
        root_node = self.build()

        if symtable is None:
            symtable = self.symtable

        if symtable is None:
            raise RuntimeError('No symbol table provided for evaluation')

        return root_node.eval(symtable)

    # criterion -> evaluable_item { uneg } op evaluable_item
    @logwrapper
    def parse_criterion(self):
        negate = False

        lhs = self.parse_evaluable_item()

        token, val = self.tokenizer.peek()

        if token == 'UNEG' or token == 'OP':
            token, val = self.tokenizer.scan()

            if token == 'UNEG':
                negate = True
                token, val = self.tokenizer.scan()

            if token != 'OP':
                raise SyntaxError('Expecting {UNEG} BOOL | OP')

            op = val
            rhs = self.parse_evaluable_item()
            return Node(lhs, op, rhs, negate)
        else:
            return lhs

    @logwrapper
    def parse_array(self):
        array_value = []

        token, val = self.tokenizer.scan()
        assert(token == 'OPENBRACKET')

        while(True):
            token_next, val_next = self.tokenizer.peek()

            if token_next == 'CLOSEBRACKET':
                self.tokenizer.scan()
                return Node(array_value, 'ARRAY', None)

            new_node = self.parse_expr()

            self.logger.debug('Appending value of type %s' % (new_node.op,))
            array_value.append(new_node)

            next_token, next_val = self.tokenizer.peek()
            if next_token == 'COMMA':
                self.tokenizer.scan()  # eat the comma
                continue

            if next_token != 'CLOSEBRACKET':
                raise SyntaxError('Expecting "]" or ","')

    # evalable_item -> function(evalable_item, ...) | identifier |
    # value
    @logwrapper
    def parse_evaluable_item(self):
        next_token, next_val = self.tokenizer.peek()
        if next_token == 'OPENBRACKET':
            return self.parse_array()

        token, val = self.tokenizer.scan()

        if token == 'NUMBER':
            return Node(int(val), 'NUMBER', None)

        if token == 'STRING':
            return Node(str(val), 'STRING', None)

        if token == 'BOOL':
            return Node(val, 'BOOL', None)

        if token == 'NONE':
            return Node(None, 'NONE', None)

        if token == 'IDENTIFIER':
            next_token, next_val = self.tokenizer.peek()
            if next_token != 'OPENPAREN':
                return Node(str(val), 'IDENTIFIER', None)
            else:
                self.tokenizer.scan()  # eat the paren

                done = False
                args = []
                function_name = str(val)

                while not done:
                    args.append(self.parse_evaluable_item())

                    token, val = self.tokenizer.scan()

                    if token == 'CLOSEPAREN':
                        # done parsing evaluable item
                        return Node(function_name, 'FUNCTION', args)

                    if token != 'COMMA':
                        raise SyntaxError('expecting comma or close paren')

        raise SyntaxError('expecting evaluable item in "%s"' %
                          self.input_expression)

    # expr -> T_OPENPAREN andexpr T_CLOSEPAREN | criterion
    @logwrapper
    def parse_expr(self):
        token, val = self.tokenizer.peek()

        if token == 'OPENPAREN':
            self.tokenizer.scan()
            node = self.parse_andexpr()
            token, val = self.tokenizer.scan()
            if token != 'CLOSEPAREN':
                raise SyntaxError('expecting close paren')
            return node
        else:
            return self.parse_criterion()

    # orexpr -> expr { T_OR expr }
    @logwrapper
    def parse_orexpr(self):
        node = self.parse_expr()

        token, val = self.tokenizer.peek()
        if token == 'OR' or (token == 'ARITH_OP' and val in ['*', '/']):
            self.tokenizer.scan()  # eat the token
            rhs = self.parse_orexpr()
            return Node(node, val.upper(), rhs)
        else:
            return node

    # andexpr -> orexpr { T_AND orexpr }
    @logwrapper
    def parse_andexpr(self):
        node = self.parse_orexpr()

        token, val = self.tokenizer.peek()
        if token == 'AND' or (token == 'ARITH_OP' and val in ['+', '-']):
            self.tokenizer.scan()  # eat the token
            rhs = self.parse_andexpr()
            return Node(node, val.upper(), rhs)
        else:
            return node

    # phrase -> andexpr EOF
    @logwrapper
    def parse_phrase(self):
        token, val = self.tokenizer.peek()

        node = self.parse_andexpr()

        token, val = self.tokenizer.scan()
        if token != 'EOF':
            raise SyntaxError('expecting EOF')

        return node


class Node(object):
    def __init__(self, lhs, op, rhs, negate=False):
        self.lhs = lhs
        self.rhs = rhs
        self.op = op
        self.negate = negate
        classname = self.__class__.__name__.lower()
        self.logger = logging.getLogger('%s.%s' % (__name__, classname))

        self.dotty_shape = 'ellipse'
        self.dotty_label = str(op)

    def maybe_dotty(self, fd, node):
        if isinstance(node, Node):
            return node.dotty(fd)

        if node is None:
            return None

        print >>fd, '"%s" -> "%s"' % (id(node), node)
        return id(node)

    def dotty(self, fd):
        left_entity = self.maybe_dotty(fd, self.lhs)
        right_entity = self.maybe_dotty(fd, self.lhs)

        if left_entity is not None:
            print >>fd, '"%s" -> "%s"' % (id(self), left_entity)

        if right_entity is not None:
            print >>fd, '"%s" -> "%s"' % (id(self), right_entity)

        print >>fd, '"%s" [label="%s" shape="%s"]' % (
            id(self), self.dotty_label, self.dotty_shape)

        return id(self)


class MapBasedOp(Node):
    def __init__(self, lhs, op, rhs):
        super(ArithOp, self).__init__(lhs, op, rhs)
        self.dotty_shape = "circle"
        self.dotty_label = self.op

    def eval(self, symtable):
        if not self.op in self.op_map:
            raise SyntaxError('Invalid op: "%s"' % self.op)

        return self.op_map[self.op](self.lhs.eval(symtable),
                                    self.rhs.eval(symtable))


class ArithOp(MapBasedOp):
    def __init__(self, lhs, op, rhs):
        super(ArithOp, self).__init__(lhs, op, rhs)
        self.op_map = {'+': operator.add,
                       '-': operator.sub,
                       '*': operator.mul,
                       '/': operator.div}


class LogicalOp(MapBasedOp):
    def __init__(self, lhs, op, rhs):
        super(ArithOp, self).__init__(lhs, op, rhs)
        self.op_map =  {'and': operator.and_,
                        'or': operator.or_}


class BoolOp(MapBasedOp):
    """
    >, <, etc
    """
    def __init__(self, lhs, op, rhs):
        super(BoolOp, self).__init__(lhs, op, rhs)
        self.op_map = {'>': operator.gt,
                       '<': operator.lt,
                       '<=': operator.le,
                       '>=': operator.ge,
                       '==': operator.eq}

class Literal(Node):
    def __init__(self, ltype, lvalue):
        super(Literal, self).__init__(lvalue, ltype, None)
        self.dotty_shape = "square"
        self.dotty_label = str(self.lhs)

    def dotty(self, fd):
        print >>fd, '"%s" [label="%s" shape="%s"]' % (
            id(self), self.dotty_label, self.dotty_shape)
        return self.id

    def eval(self, symtable):
        return self.lhs


class Identifier(Node):
    def __init__(self, ivalue):
        super(Literal, self).__init__(ivalue, 'IDENTIFIER', None)
        self.dotty_shape = "square"
        self.dotty_label = str(self.lhs)

    def eval(self, symtable):
        if not self.lhs in symtable:
            raise SyntaxError('Unknown identifier: "%s"' % self.lhs)
        return symtable[self.lhs]

    # def arithop(self, op, val1, val2):
    #     if op == '+':
    #         return val1 + val2
    #     if op == '-':
    #         return val1 - val2
    #     if op == '*':
    #         return val1 * val2
    #     if op == '/':
    #         return val1 / val2
    #     raise SyntaxError('Bad arith op: %s' % op)

    # def emit(self, indent=0):
    #     outstr = ' ' * indent
    #     outstr += self.op + ': '

    #     if self.op in ['NUMBER', 'BOOL', 'NONE',
    #                    'STRING', 'IDENTIFIER']:
    #         outstr += self.value_to_s()
    #     elif self.op in ['+', '-', '*', '/']:
    #         outstr += '\n%s%s' % (self.lhs.emit(indent + 1),
    #                               self.rhs.emit(indent + 1))
    #     elif self.op in ['AND', 'OR']:
    #         outstr += '\n%s%s' % (self.lhs.emit(indent + 1),
    #                               self.rhs.emit(indent + 1))
    #     else:
    #         raise SyntaxError('Non emittable op: %s' % (self.op,))

    #     return outstr + '\n'

    # def concrete(self, ns):
    #     if self.op in ['NUMBER', 'BOOL', 'NONE']:
    #         return self.value_to_s()

    #     if self.op in ['STRING', 'IDENTIFIER']:
    #         string = self.canonicalize_string(self.lhs, ns)
    #         if self.op == 'STRING':
    #             string = "'%s'" % string.replace("'", "\'")
    #         return string

    #     if self.op == 'FUNCTION':
    #         return '%s(%s)' % (
    #             self.lhs, ', '.join(map(lambda x: x.concrete(ns),
    #                                     self.rhs)))

    #     if self.op == 'AND' or self.op == 'OR':
    #         return '(%s) %s (%s)' % (self.lhs.concrete(ns),
    #                                  self.op,
    #                                  self.rhs.concrete(ns))

    #     return '%s %s%s %s' % (self.lhs.concrete(ns),
    #                            '!' if self.negate else '',
    #                            self.op,
    #                            self.rhs.concrete(ns))

    # def canonicalize_string(self, string, ns):
    #     result = string

    #     match = re.match("(.*)\{(.*?)}(.*)", string)
    #     if match is not None:
    #         start = match.group(1)
    #         term = match.group(2)
    #         end = match.group(3)

    #         if term in ns:
    #             result = "%s%s%s" % (start,
    #                                  ns[term],
    #                                  self.canonicalize_string(end, ns))
    #         else:
    #             result = "%s{%s}%s" % (start,
    #                                    term,
    #                                    self.canonicalize_string(end, ns))

    #     return result

    # def value_to_s(self):
    #     if self.op == 'STRING':
    #         return "'%s'" % self.lhs.replace("\\'", "'").replace("'", "\\'")

    #     if self.op == 'BOOL':
    #         return str(self.lhs).lower()

    #     return str(self.lhs)

    # def to_s(self):
    #     if self.op in ['NUMBER', 'BOOL', 'STRING',
    #                    'IDENTIFIER', 'NONE']:
    #         return self.value_to_s()

    #     if self.op == 'FUNCTION':
    #         return '%s(%s)' % (self.lhs, ', '. join(map(lambda x: x.to_s(),
    #                                                     self.rhs)))

    #     if self.op == 'AND' or self.op == 'OR':
    #         return '(%s) %s (%s)' % (self.lhs.to_s(), self.op, self.rhs.to_s())

    #     return '%s %s%s %s' % (self.lhs.to_s(), '!' if self.negate else '',
    #                            self.op.lower(), self.rhs.to_s())

    # def canonicalize_identifier(self, identifier, ns=None):
    #     if not identifier:
    #         return None

    #     if ns is None:
    #         ns = {}

    #     # check for string interpolation in identifier.
    #     match = re.match("(.*)\{(.*?)}(.*)", identifier)
    #     if match is not None:
    #         resolved_match_term = self.eval_identifier(match.group(2), ns)
    #         new_identifier = "%s%s%s" % (match.group(1), resolved_match_term,
    #                                      match.group(3))

    #         return self.canonicalize_identifier(new_identifier, ns)

    #     return identifier

    # def assign_identifier(self, node, identifier, value, ns=None):
    #     # there is lots of strange flakeyness here.

    #     self.logger.debug('assigning id using api: %s' % self.api)

    #     if not identifier:
    #         return None

    #     self.logger.debug('setting %s to %s' % (identifier, value))
    #     canonical = self.canonicalize_identifier(node, identifier,
    #                                              symbol_table)

    #     self.logger.debug('canonicalized %s to %s' % (identifier, canonical))
    #     if canonical.find('.') == -1:
    #         # do an update on this node.
    #         self.api._model_update_by_id(object_type, node['id'],
    #                                      {canonical: value})
    #         return
    #     else:
    #         (attr, rest) = canonical.split('.', 1)

    #         self.logger.debug('attr: %s, object_type: %s' %
    #                           (attr, object_type))

    #         if attr == 'facts' and object_type == 'nodes':
    #             existing_fact = self.api._model_query(
    #                 'facts',
    #                 'node_id=%d and key=%s' % (node['id'], rest))

    #             if existing_fact:
    #                 self.api._model_update_by_id('facts',
    #                                              existing_fact['id'],
    #                                              {'value': value})
    #             else:
    #                 self.api._model_create('facts', {'node_id': node['id'],
    #                                                  'key': rest,
    #                                                  'value': value})
    #         elif attr == 'attrs' and object_type == 'nodes':
    #             existing_attr = self.api._model_query(
    #                 'attrs',
    #                 'node_id=%d and key=%s' % (node['id'], rest))

    #             if existing_attr:
    #                 self.api._model_update_by_id('attrs',
    #                                              existing_attr['id'],
    #                                              {'value': value})
    #             else:
    #                 self.api._model_create('attrs', {'node_id': node['id'],
    #                                                  'key': rest,
    #                                                  'value': value})
    #         return

    #     raise ValueError('Cannot express assignment to id: %s' % identifier)

    # def eval_identifier(self, identifier, ns=None):
    #     self.logger.debug('resolving identifier "%s" with ns %s' %
    #                       (identifier, ns))

    #     if identifier in ns:
    #         return ns[identifier]

    #     return None

    # def __str__(self):
    #     if self.op == 'STRING':
    #         return str(self.lhs)

    #     if self.op == 'NUMBER':
    #         return str(int(self.lhs))

    #     if self.op == 'BOOL':
    #         return str(self.lhs)

    #     if self.op == 'IDENTIFIER':
    #         return 'IDENTIFIER %s' % self.lhs

    #     if self.op == 'NONE':
    #         return 'VALUE None'

    #     if self.op == 'FUNCTION':
    #         return 'FN %s(%s)' % (str(self.lhs), ', '.join(map(str, self.rhs)))

    #     return '(%s) %s (%s)' % (str(self.lhs), self.op, str(self.rhs))

    # def eval_node(self, functions=None, ns=None):
    #     rhs_val = None
    #     lhs_val = None
    #     result = False

    #     retval = None

    #     if ns is None:
    #         ns = {}

    #     if functions is None:
    #         functions = default_functions

    #     self.logger.debug('evaluating %s with ns %s' %
    #                       (str(self), ns))

    #     if self.op in ['STRING', 'NUMBER', 'BOOL',
    #                    'IDENTIFIER', 'FUNCTION', 'ARRAY', 'NONE']:
    #         if self.op == 'STRING':
    #             # check for string interpolation in identifier.
    #             retval = str(self.lhs)
    #             match = re.match("(.*)\{(.*?)}(.*)", retval)
    #             if match is not None:
    #                 resolved_match_term = self.eval_identifier(
    #                     match.group(2), ns)
    #                 retval = "%s%s%s" % (match.group(1), resolved_match_term,
    #                                      match.group(3))

    #         if self.op == 'NUMBER':
    #             retval = int(self.lhs)

    #         if self.op == 'BOOL':
    #             if self.lhs == 'TRUE':
    #                 retval = True
    #             else:
    #                 retval = False

    #         if self.op == 'IDENTIFIER':
    #             retval = self.eval_identifier(self.lhs, ns)

    #         if self.op == 'NONE':
    #             retval = None

    #         if self.op == 'ARRAY':
    #             retval = []
    #             for item in self.lhs:
    #                 retval.append(item.eval_node(ns=ns))
    #             return retval

    #         if self.op == 'FUNCTION':
    #             if not self.lhs in functions:
    #                 raise SyntaxError('unknown function %s' % self.lhs)

    #             args = map(lambda x: x.eval_node(functions, ns),
    #                        self.rhs)

    #             retval = functions[self.lhs](*args)

    #         self.logger.debug('evaluated %s to %s' % (str(self), retval))
    #         return retval

    #     self.logger.debug('arithmetic op, type %s' % self.op)

    #     # otherwise arithmetic op
    #     lhs_val = self.lhs.eval_node(functions, ns)
    #     rhs_val = self.rhs.eval_node(functions, ns)

    #     # wrong types is always false
    #     if type(lhs_val) == unicode:
    #         lhs_val = str(lhs_val)

    #     if type(rhs_val) == unicode:
    #         rhs_val = str(rhs_val)

    #     self.logger.debug('checking %s %s %s' % (lhs_val, self.op, rhs_val))

    #     # handle arith first
    #     if self.op in ['+', '-', '/', '*']:
    #         if not isinstance(lhs_val, type(rhs_val)):
    #             raise SyntaxError('Mismatched types for "%s"' % self.op)

    #         result = self.arithop(self.op, lhs_val, rhs_val)
    #         return result

    #     if self.op == '=':
    #         if lhs_val == rhs_val:
    #             result = True
    #     elif self.op == '<':
    #         if type(lhs_val) != type(rhs_val):
    #             return False
    #         elif lhs_val < rhs_val:
    #             result = True
    #     elif self.op == '>':
    #         if type(lhs_val) != type(rhs_val):
    #             return False
    #         elif lhs_val > rhs_val:
    #             result = True
    #     elif self.op == '<=':
    #         if type(lhs_val) != type(rhs_val):
    #             return False
    #         elif lhs_val <= rhs_val:
    #             result = True
    #     elif self.op == '>=':
    #         if type(lhs_val) != type(rhs_val):
    #             return False
    #         elif lhs_val >= rhs_val:
    #             result = True
    #     elif self.op == 'AND':
    #         result = lhs_val and rhs_val
    #     elif self.op == 'OR':
    #         result = lhs_val or rhs_val
    #     elif self.op == 'IN':
    #         try:
    #             if lhs_val in rhs_val:
    #                 result = True
    #         except Exception:
    #             result = False
    #     elif self.op == ':=':
    #         if self.lhs.op != 'IDENTIFIER':
    #             raise SyntaxError('must assign to identifier: %s' %
    #                               (self.lhs.lhs))
    #         self.logger.debug('setting %s to %s' % (self.lhs.lhs, rhs_val))
    #         self.assign_identifier(self.lhs.lhs, rhs_val, ns)
    #         result = rhs_val
    #     else:
    #         raise SyntaxError('bad op token (%s)' % self.op)

    #     if self.negate:
    #         result = not result

    #     self.logger.debug('Returning %s' % (result,))
    #     return result
