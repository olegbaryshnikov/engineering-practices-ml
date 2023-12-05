flake8:
```
./hw1/test.py:4:1: E302 expected 2 blank lines, found 1
./hw1/test.py:15:80: E501 line too long (90 > 79 characters)
./hw1/test.py:21:80: E501 line too long (85 > 79 characters)
./hw1/test.py:24:80: E501 line too long (103 > 79 characters)
./hw1/test.py:27:1: W293 blank line contains whitespace
./hw1/test.py:57:1: W293 blank line contains whitespace
./hw1/test.py:58:5: E303 too many blank lines (2)
```

black:
```
--- hw1/test.py 2023-12-05 13:00:08+00:00
+++ hw1/test.py 2023-12-05 13:11:04.901932+00:00
@@ -1,20 +1,23 @@
 # ChatGPT generated code example
 import numpy as np
 
-class myClass():
+
+class myClass:
     def __init__(self, initial_val=0):
         self.myAttribute = initial_val
 
     def calculateSum(self, firstNumber, secondNumber):
         Result = firstNumber + secondNumber
         return Result
 
     def PrintSum(self):
         print("Result:", self.calculateSum(10, 20))
 
-    def MethodWithLongNameAndManyParameters(self, param1, param2, param3, param4, param5):
+    def MethodWithLongNameAndManyParameters(
+        self, param1, param2, param3, param4, param5
+    ):
         # This method intentionally violates PEP8's line length recommendation
         result = param1 + param2 + param3 + param4 + param5
         return result
 
     def another_method(self):
@@ -22,11 +25,11 @@
 
     def yet_another_method(self):
         # This method violates PEP8's recommendation for blank lines around function/method definitions
         result = self.calculateSum(30, 40)
         return result
-    
+
     def mul(self, x, y):
         return np.matmul(x, y)
 
 
 if __name__ == "__main__":
@@ -52,8 +55,7 @@
             self.AnotherAttribute = "example"
 
         def AnotherMethod(self):
             print("This method intentionally violates PEP8 conventions.")
 
-    
     another_instance = AnotherClass()
     another_instance.AnotherMethod()
would reformat hw1/test.py

All done! ✨ 🍰 ✨
1 file would be reformatted.
```

isort:
```
ERROR: /ep-ml/hw1/hw1/test.py Imports are incorrectly sorted and/or formatted.
```


