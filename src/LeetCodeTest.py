# 729. My Calendar I
# ou are implementing a program to use as your calendar. We can add a new event if adding the event will not cause a double booking.
class MyCalendar:

    def __init__(self):
        self.starts = []
        self.ends = []

    def book(self, startTime: int, endTime: int) -> bool:
        def binsearchleft(arr, time, isleft):
            n = len(arr)
            left, right = 0, n
            if isleft:
                while left < right:
                    mid = (left+right)//2
                    if arr[mid] < time:
                        left = mid + 1
                    else:
                        right = mid
                return left
            else:
                while left < right:
                    mid = (left+right)//2
                    if arr[mid] <= time:
                        left = mid + 1
                    else:
                        right = mid
                return left
        idxstart = binsearchleft(self.starts,endTime,True)
        idxend = binsearchleft(self.ends,startTime,False)
        if idxstart == idxend:
            self.starts.insert(idxstart, startTime)
            self.ends.insert(idxend,endTime)
            return True
        return False


# 731. My Calendar II
from sortedcontainers import SortedDict
class MyCalendarTwo:

    def __init__(self):
        self.DictCount = SortedDict()
        self.max_overlap = 2

    def book(self, startTime: int, endTime: int) -> bool:
        self.DictCount[startTime] = self.DictCount.get(startTime,0) + 1
        self.DictCount[endTime] = self.DictCount.get(endTime, 0) - 1

        current_overlap = 0
        for count in self.DictCount.values():
            current_overlap += count
            if current_overlap > self.max_overlap:
                # Revert the changes if overlap exceeds max allowed
                self.DictCount[startTime] -= 1
                self.DictCount[endTime] += 1
                if self.DictCount[startTime] == 0:
                    del self.DictCount[startTime]
                return False
        return True
        

# LC 732
class MyCalendarThree:

    def __init__(self):
        self.DictCount = SortedDict()

    def book(self, startTime: int, endTime: int) -> bool:
        self.DictCount[startTime] = self.DictCount.get(startTime,0) + 1
        self.DictCount[endTime] = self.DictCount.get(endTime, 0) - 1

        current_overlap = 0
        res = 0
        for count in self.DictCount.values():
            current_overlap += count
            res = max(res,current_overlap)
        return res

# 224. Basic Calculator
# Given a string s representing a valid expression, implement a basic calculator 
# to evaluate it, and return the result of the evaluation.
# Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        current_number = 0
        current_sign = 1
        result = 0
        for char in s:
            if char.isdigit():
                current_number = current_number * 10 + int(char)
            elif char in "+-":
                result += current_sign * current_number
                current_number = 0
                current_sign = 1 if char == '+' else -1
            elif char == '(':
                stack.append(result)
                stack.append(current_sign)
                result = 0
                current_sign = 1
            elif char == ')':
                result += current_sign * current_number
                result *= stack.pop() # sign before the parenthesis
                result += stack.pop() # result calculated before the parenthesis
                current_number = 0
        if current_number != 0:
            result += current_sign * current_number
        
        return result

# 227. Basic Calculator II
# Given a string s which represents an expression, evaluate this expression and return its value. 
# The integer division should truncate toward zero.
# You may assume that the given expression is always valid. All intermediate results will be in the range of [-231, 231 - 1].
# Input: s = "3+2*2"
# Output: 7
# Example 2:
# Input: s = " 3/2 "
# Output: 1
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        current_number = 0
        prev_op = '+'
        s += '+'

        for char in s:
            if char.isdigit():
                current_number = current_number * 10 + int(char)
            elif char == ' ':
                continue
            else:
                if prev_op == '+':
                    stack.append(current_number)
                elif prev_op == '-':
                    stack.append(-current_number)
                elif prev_op == '*':
                    temp = stack.pop()
                    stack.append(current_number*temp)
                elif prev_op == '/':
                    temp = stack.pop()
                    stack.append(int(temp/current_number))
                current_number = 0
                prev_op = char
        
        return sum(stack)

# 772. Basic Calculator III
# Implement a basic calculator to evaluate a simple expression string.
# The expression string contains only non-negative integers, '+', '-', '*', '/' operators, and open '(' and closing parentheses ')'. The integer division should truncate toward zero.
class Solution:
    def calculate(self, s: str) -> int:
        def evaluate(x, y, operator):
            if operator == '+':
                return x
            if operator == '-':
                return -x
            if operator == '*':
                return x * y
            return int(x / y)
        
        stack = []
        curr = 0
        prev_op = '+'
        s += '+'
        for char in s:
            if char == ' ':
                continue
            if char.isdigit():
                curr = curr * 10 + int(char)
            elif char == '(':
                stack.append(prev_op)
                prev_op = '+'
            else:
                if prev_op in '*/':
                    stack.append(evaluate(stack.pop(), curr,prev_op))
                else:
                    stack.append(evaluate(curr, 0, prev_op))
                curr = 0
                prev_op = char
                if char == ')':
                    while type(stack[-1]) == int:
                        curr += stack.pop()
                    prev_op = stack.pop()
        return sum(stack)
    
# 770. Basic Calculator IV
# Given an expression such as expression = "e + 8 - a + 5" and an evaluation map such as {"e": 1} (given in terms of evalvars = ["e"] and evalints = [1]), return a list of tokens representing the simplified expression, such as ["-1*a","14"]
# An expression alternates chunks and symbols, with a space separating each chunk and symbol.
# chunk is either an expression in parentheses, a variable, or a non-negative integer.
#A variable is a string of lowercase letters (not including digits.) Note that variables can be multiple letters, and note that variables never have a leading coefficient or unary operator like "2x" or "-x".
#Expressions are evaluated in the usual order: brackets first, then multiplication, then addition and subtraction.

#For example, expression = "1 + 2 * 3" has an answer of ["7"].
#The format of the output is as follows:

#For each term of free variables with a non-zero coefficient, we write the free variables within a term in sorted order lexicographically.
#For example, we would never write a term like "b*a*c", only "a*b*c".
#Terms have degrees equal to the number of free variables being multiplied, counting multiplicity. We write the largest degree terms of our answer first, breaking ties by lexicographic order ignoring the leading coefficient of the term.
#For example, "a*a*b*c" has degree 4.
#The leading coefficient of the term is placed directly to the left with an asterisk separating it from the variables (if they exist.) A leading coefficient of 1 is still printed.
#An example of a well-formatted answer is ["-2*a*a*a", "3*a*a*b", "3*b*b", "4*a", "5*c", "-6"].
#Terms (including constant terms) with coefficient 0 are not included.
#For example, an expression of "0" has an output of [].
class Expression:
    def __init__(self, value: Dict[Tuple[str, ...], int] = None):
        self.value = value if value else {}

    def __sub__(self, other: 'Expression') -> 'Expression':
        res = dict(self.value)

        for var, coef in other.value.items():
            res[var] = res.get(var, 0) - coef

        return Expression(value=res)

    def __add__(self, other: 'Expression') -> 'Expression':
        res = dict(self.value)

        for var, coef in other.value.items():
            res[var] = res.get(var, 0) + coef

        return Expression(value=res)

    def __mul__(self, other: 'Expression') -> 'Expression':
        res = {}
        for var1, coef1 in self.value.items():
            for var2, coef2 in other.value.items():
                new_var = tuple(sorted(var1 + var2))
                res[new_var] = res.get(new_var, 0) + coef1 * coef2
        return Expression(value=res)

    def __neg__(self) -> 'Expression':
        return Expression() - self

class Solution:
    def basicCalculatorIV(self, expression: str, evalvars: List[str], evalints: List[int]) -> List[str]:
        varmap = dict(zip(evalvars, evalints))

        def tokenize(expression: str) -> List[str]:
            return re.findall(r'[a-z]+|[0-9]+|[\+\-\*\(\)]', expression)

        def process_tokens(tokens: List[str]) -> Expression:
            stack = []
            xpr = Expression()
            last_op = '+'
            ops = {'+', '-', '*'}

            def apply_op(op: str, expr: Expression) -> None:
                if op == '+':
                    stack.append(expr)
                elif op == '-':
                    stack.append(-expr)
                elif op == '*':
                    stack.append(stack.pop() * expr)

            for token in tokens:
                if token.isdigit():
                    xpr = Expression(value={(): int(token)})
                elif token.isalpha():
                    if token in varmap:
                        xpr = Expression(value={(): varmap[token]})
                    else:
                        xpr = Expression(value={(token,): 1})
                elif token in ops:
                    apply_op(last_op, xpr)
                    xpr = Expression()
                    last_op = token
                elif token == '(':
                    stack.append(last_op)
                    last_op = '+'
                elif token == ')':
                    apply_op(last_op, xpr)
                    xpr = Expression()

                    while isinstance(stack[-1], Expression):
                        xpr += stack.pop()

                    last_op = stack.pop()

            apply_op(last_op, xpr)
            return sum(stack, Expression())

        def format_result(result: Expression) -> List[str]:
            formatted = []

            for var, coef in sorted(
                result.value.items(),
                key=lambda x: (-len(x[0]), x[0])):
                if coef:
                    formatted.append(f'{coef}' + ('*' + '*'.join(var) if var else ''))

            return formatted

        tokens = tokenize(expression)
        result = process_tokens(tokens)

        return format_result(result)
    



    