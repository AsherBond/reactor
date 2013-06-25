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

import unittest2
import reactor.ast


class AstTests(unittest2.TestCase):
    def setUp(self):
        self.s1 = {'strfield': 'testing',
                   'intfield': 3,
                   'arrayfield': [1, 2, 3]}

    def tearDown(self):
        pass

    def _eval(self, symtable, expression):
        ast = reactor.ast.AstBuilder(input_expression=expression,
                                     symtable=symtable)
        return ast.eval()

    def test_int_equality(self):
        self.assertTrue(self._eval(self.s1, 'intfield == 3'))
        self.assertFalse(self._eval(self.s1, 'intfield == 2'))

        self.assertFalse(self._eval(self.s1, 'intfield != 3'))
        self.assertTrue(self._eval(self.s1, 'intfield != 2'))

    def test_int_comparison(self):
        self.assertTrue(self._eval(self.s1, 'intfield > 2'))
        self.assertTrue(self._eval(self.s1, 'intfield < 4'))
        self.assertFalse(self._eval(self.s1, 'intfield > 4'))
        self.assertFalse(self._eval(self.s1, 'intfield < 2'))

    def test_str_equality(self):
        self.assertTrue(self._eval(self.s1, 'strfield == "testing"'))
        self.assertFalse(self._eval(self.s1, 'strfield == "x"'))
        self.assertFalse(self._eval(self.s1, 'strfield != "testing"'))
        self.assertTrue(self._eval(self.s1, 'strfield != "x"'))

    def test_str_substring(self):
        self.assertTrue(self._eval(self.s1, '"test" in strfield'))
        self.assertFalse(self._eval(self.s1, '"x" in strfield'))

        # need not function
        # self.assertFalse(self._eval(self.s1, '"test" !in strfield'))
        # self.assertTrue(self._eval(self.s1, '"x" !in strfield'))

    # no array equality... no static arrays
    def test_len(self):
        self.assertTrue(self._eval(self.s1, 'count(arrayfield) == 3'))
        # Nones for invalid types
        self.assertEqual(self._eval(self.s1, 'count(intfield)'), None)

    def test_arith(self):
        self.assertEqual(self._eval(self.s1,
                                    '3 + 5'), 8)
        self.assertEqual(self._eval(self.s1,
                                    '3 * 5'), 15)
        self.assertEqual(self._eval(self.s1,
                                    '8 / 2'), 4)
        self.assertEqual(self._eval(self.s1,
                                    '7 / 2'), 3)
        self.assertEqual(self._eval(self.s1,
                                    '8 - 2'), 6)

    def test_precedence(self):
        self.assertEqual(self._eval(self.s1,
                                    '1 + 2 * 3'), 7)
        self.assertEqual(self._eval(self.s1,
                                    '3 * 2 + 1'), 7)
        self.assertTrue(self._eval(self.s1,
                                   'true and false or true'))
        self.assertTrue(self._eval(self.s1,
                                   'true or false and true'))

    # def test_assignment(self):
    #     ns = {}
    #     self._eval(self.s1, 'arf := 3')

    #     self.assertEqual(ns['arf'], 3)
