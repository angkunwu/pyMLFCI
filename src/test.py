# Virtu Financial OA
# Task2 LC 1196
# A box with capacity of 5000 grams.
# Write a function that given a zero-indexed
# array A of N integers, representing
# the weight of items already in the box and 
# each apple's weight. Return the maximum
# number of apples that could fit in the box.
# The first element of the array is the weight
# of the items already in the box, and the rest
# of the elements are the weights of the apples.
def Solution(A:List[int])->int:
    RemainW = 5000 - A[0]
    A = A[1:]
    A.sort()
    count = 0
    for i in A:
        if RemainW - i >= 0:
            RemainW -= i
            count += 1
        else:
            break
    return count

# Task 2
# Given a string S, return an integer that
# represents the number of ways in which
# we can select non-empty substrings of S
# where all characters of the substring are identical
# Two substrings with the same letters but different
# locations are still considered different
# For example, for S = "zzzyz", the function should
# return 8, as the following substrings are possible:
# z, z, z, z, zz, zz, zzz, y
def Solution(S:str)->int:
    count = 0
    for i in range(len(S)):
        j = i + 1
        repeat = 1
        while j < len(S) and S[j] == S[i]:
            repeat += 1
            j += 1
        count += repeat * (repeat + 1) // 2
    return count



# Task 3
#  Given a string S encoding a list of a decimal
# Integer N, return a string representing the 
# Hexspeak representation H of N if H is a 
# valid Hexspeak word, otherwise return "ERROR"
def Hexspeak(S:str)->str:
    N = int(S)
    #Hex = hex(N)[2:]
    Hex = ""
    hexstr = "0123456789ABCDEF"
    while N:
        residue = N % 16
        char = hexstr[residue]
        Hex = char + Hex
        N = N // 16
    Hex = Hex.replace('0', 'O')
    Hex = Hex.replace('1', 'I')
    for i in Hex:
        if i not in 'ABCDEFIO':
            return 'ERROR'
    return Hex.upper()
# Task 4
# Given an integer X, returns an integer that
# corresponds to the minimum number of steps
# required to change X to a Fibonacci number
def Fibonacci(X:int)->int:
    fib1 = 0
    fib2 = 1
    while fib2 < X:
        fib1, fib2 = fib2, fib1 + fib2
    if fib2 == X:
        return 0
    else:    
        return min(X - fib1, fib2 - X)

# Task 5
# Given a zero-indexed array A consisting of N
# integers, representing the initial test scores of
# a row of students, returns an array of integers
# representing the final test scores (in the same order)
# there is a group of students who sit next to each other in a row. Each day, the students study together and take a test at the end of the day. Tests scores for a given student can only change once per day as follows:
#if a student sits immediately between two students with better scores, that student's score will improve by 1 when they take the next test.
#If a student sits between two students with worse scores, that student's test score will decrease by 1.
#This process will repeat each day as long as at least one student's score changes. Note that the first and last student in the row never change their scores as they never sit between two students.
#Return an array representing the final test scores for each student.
def DynamicScores(A:List[int])->List[int]:
    while True:
        change = False
        newA = A[:]
        for i in range(1, len(A) - 1):
            if A[i] < A[i - 1] and A[i] < A[i + 1]:
                newA[i] += 1
                change = True
            elif A[i] > A[i - 1] and A[i] > A[i + 1]:
                newA[i] -= 1
                change = True
        if not change:
            break
        A = newA
    return A

def CountNum(n):
    res = 0
    for num in range(2*10**(n-1), 10**n):
        strnum = str(num)
        if "911" in strnum:
            continue
        res += 1
    return res

CountNum(3) #799
CountNum(4) #7982
CountNum(5) #79740
CountNum(6) #796601
CountNum(7) #7958028