# ChatGPT generated code example
import numpy as np


class myClass():
    def __init__(self, initial_val=0):
        self.myAttribute = initial_val

    def calculateSum(self, firstNumber, secondNumber):
        Result = firstNumber + secondNumber
        return Result

    def PrintSum(self):
        print("Result:", self.calculateSum(10, 20))

    def MethodWithLongNameAndManyParameters(self, param1, param2, param3, param4, param5):
        # This method intentionally violates PEP8's line length recommendation
        result = param1 + param2 + param3 + param4 + param5
        return result

    def another_method(self):
        print("This method has a single-line definition violating PEP8 conventions.")

    def yet_another_method(self):
        # This method violates PEP8's recommendation for blank lines around function/method definitions
        result = self.calculateSum(30, 40)
        return result
    
    def mul(self, x, y):
        return np.matmul(x, y)


if __name__ == "__main__":
    instance = myClass()
    instance.PrintSum()

    long_variable_name_with_multiple_words = 42
    anotherLongVariable = "example"

    def function_with_mixed_case_and_long_name():
        print("This function intentionally violates PEP8 conventions.")

    function_with_mixed_case_and_long_name()

    def function_with_docstring():
        """This docstring intentionally violates PEP8 conventions."""
        print("This function has a violating docstring.")

    function_with_docstring()

    class AnotherClass:
        def __init__(self):
            self.AnotherAttribute = "example"

        def AnotherMethod(self):
            print("This method intentionally violates PEP8 conventions.")

    
    another_instance = AnotherClass()
    another_instance.AnotherMethod()
