###Intoduction
#Say "Hello, World!" With Python
if __name__ == '__main__':
    print "Hello, World!"
    
#Arithmetic Operators
if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(a+b)
    print(a-b)
    print(a*b)
#Python:Division
from __future__ import division

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
    print(a//b)
    print(a/b)
#Loops
if __name__ == '__main__':
    n = int(raw_input())
    for i in range(n):
        print(i**2)
#Write a function
def is_leap(year):
    leap = False
    # Write your logic here
    if(year%4==0 and year%100!=0 or year%400==0):
        leap=True
    
    return leap
#Python If-Else
#!/bin/python

import math
import os
import random
import re
import sys
if __name__ == '__main__':
    n = int(raw_input().strip())
    if n%2==1:
        print('Weird')
    else:
        if n<=5 and n>=2:
            print('Not Weird')
        elif n<=20 and n>=6:
            print('Weird')
        else:
            print('Not Weird')
#Print function
from __future__ import print_function

if __name__ == '__main__':
    n = int(raw_input())
    for i in range(1,n+1):
        print(i,end='')
#List Comprehensions
if __name__ == '__main__':
    x = int(raw_input())
    y = int(raw_input())
    z = int(raw_input())
    n = int(raw_input())
    tab=[]
    for i in range(x+1):
        for j in range (y+1):
            for k in range(z+1):
                if (i+j+k) != n:
                    tab.append([i,j,k])
    print(tab)                
#Nested Lists
tab=[]
sc=[]
for _ in range(int(raw_input())):
    name = raw_input()
    score = float(raw_input())
    tab.append([name,score])
    sc.append(score)
sc_sort=sorted(sc)
mini=min(sc)
i=0
while sc_sort[i]==mini:
    i=i+1
for a,b in sorted(tab):
    if b==sc_sort[i]:
        print(a)



# Exercise 11 - Basic data types - Finding the percentage
if __name__ == '__main__':
    n = int(raw_input())
    student_marks = {}
    for _ in range(n):
        line = raw_input().split()
        name, scores = line[0], line[1:]
        scores = map(float, scores)
        student_marks[name] = scores
    query_name = raw_input()
    print("{0:.2f}".format((sum(student_marks[query_name])/len(student_marks[query_name]))))


# Exercise 12 - Basic data types - Lists
if __name__ == '__main__':
    N = int(input())
    l = []
    for n in range(N):
    
        x = raw_input().split(" ")
        if x[0] == 'insert':
            l.insert(int(x[1]), int(x[2]))
        if x[0] == 'print':
            print(l)
        if x[0] == 'remove':
            l.remove(int(x[1]))
        if x[0] == 'append':
            l.append(int(x[1]))
        if x[0] == 'sort':
            l = sorted(l)
        if x[0] == 'pop':
            l.pop()
        if x[0] == 'reverse':
            l = l[::-1]
    

# Exercise 13 - Basic data types - Tuples
if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
    t=tuple(integer_list)
    print(hash(t))

# Exercise 14 - Strings - sWAP cASE
def swap_case(s):
    s1=''
    for i in range(len(s)):
        if s[i].isupper():
            s1+=s[i].lower()
        else:
            s1+=s[i].capitalize()
            

    return(s1)

# Exercise 15 - Strings - String Split and Join
def split_and_join(line):
    # write your code here
    line = line.split(" ")
    line = "-".join(line)
    return(line)

# Exercise 16 - Strings - What's Your Name?
def print_full_name(a, b):
    ch="Hello "+a+" "+b+"! You just delved into python."
    print (ch)

# Exercise 17 - Strings - Mutations
def mutate_string(string, position, character):
    return (string[:position]+character+string[(position+1):])
# Exercise 18 - Strings - Find a string
def count_substring(string, sub_string):
    res=0
    for i in range(len(string)-len(sub_string)+1):
        if string[i:i+len(sub_string)]==sub_string:
            res+=1

    return (res)

# Exercise 19 - Strings - String Validators
if __name__ == '__main__':
    s = raw_input()
    res1=False
    res2=False
    res3=False
    res4=False
    res5=False
    for i in range(len(s)):
        res1=res1 or s[i].isalnum()
        res2=res2 or s[i].isalpha()
        res3=res3 or s[i].isdigit()
        res4=res4 or s[i].islower()
        res5=res5 or s[i].isupper()
    print(res1)
    print(res2)
    print(res3)
    print(res4)
    print(res5)



# Exercise 20 - Strings - Text Alignment
#Replace all ______ with rjust, ljust or center. 

thickness = int(raw_input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print (c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1)

#Top Pillars
for i in range(thickness+1):
    print (c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)

#Middle Belt
for i in range((thickness+1)/2):
    print (c*thickness*5).center(thickness*6)    

#Bottom Pillars
for i in range(thickness+1):
    print (c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)    

#Bottom Cone
for i in range(thickness):
    print ((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6)



# Exercise 21 - Strings - Text Wrap
def wrap(string, max_width):
    i=0
    liste=[]
    while i<len(string)-max_width+1:
        liste.append(string[i:i+max_width])
        i+=max_width
    liste.append(string[i:])
    return('\n'.join(liste))


# Exercise 22 - Strings - Designer Door Mat

n,m=map(int, raw_input().split())
motif='.|.'
for i in range((n-1)//2):
    print((motif*(2*i+1)).center(m,"-"))
print('WELCOME'.center(m,"-"))
for i in range(((n-1)//2)-1,-1,-1):
    print((motif*(2*i+1)).center(m,"-"))

# Exercise 23 - Strings - String Formatting
def print_formatted(number):
    # your code goes here
    for i in range(1,number+1):

        print "{0:{width}d} {0:{width}o} {0:{width}X} {0:{width}b}".format(i, width=len(str(bin(number)[2:])))

# Exercise 24 - Strings - Alphabet Rangoli
def print_rangoli(size):
    # your code goes here
    alphabet='abcdefghijklmnopqrstuvwxyz'
    
    ligne = []
    for i in range(size):
        s = "-".join(alphabet[i:size])
        ligne.append((s[::-1]+s[1:]).center(4*size-3, "-"))
    print('\n'.join(ligne[:0:-1]+ligne))

# Exercise 25 - Strings - Capitalize!
def solve(s):
    liste=s.split(" ")
    ch=''
    for i in range(len(liste)-1):
        ch=ch+ liste[i].capitalize()+' '
    ch=ch+(liste[len(liste)-1].capitalize())
        
    return(ch)

# Exercise 26 - Strings - The Minion Game
def minion_game(string):
    # your code goes here 
    player_vow=0
    player_con=0
    list_vow=['A','E','I','O','U']
    for i in range(len(string)):
        if string[i] in list_vow:
            player_vow+=len(string)-i
    for k in range(len(string)):
        if not(string[k] in list_vow) :
            player_con+=len(string)-k

    if player_con==player_vow:
        print('Draw')
    elif player_vow<player_con:
        res='Stuart '+str(player_con)
        print(res)
    else:
        res='Kevin '+str(player_vow)
        print(res)

# Exercise 27 - Strings - Merge the Tools!
def merge_the_tools(string, k):
    # your code goes here
    list_t=[]
    for i in range(len(string)//k):
        list_t.append(string[i*k:(i+1)*k])
    
    for i in range(len(list_t)):
        ch=''
        for j in range(len(list_t[i])):  
            if not(list_t[i][j]in ch):
                ch+=list_t[i][j]
        print(ch)

# Exercise 28 - Sets - Introduction to Sets
def average(array):
    # your code goes here
    
    s=set()
    for i in range(len(array)):
        s.add(int(array[i]))
    return(sum(s)/len(s))

# Exercise 29 - Sets - No Idea!
# Enter your code here. Read input from STDIN. Print output to STDOUT
n,m= raw_input().split()
arr_initial= raw_input().split()
A=set(raw_input().split())
B=set(raw_input().split())
gain=0
for i in range(int(n)):
    if arr_initial[i] in (A):
        gain+=1
    elif arr_initial[i] in (B):
        gain=gain-1

print(gain)

# Exercise 30 - Sets - Symmetric Difference
# Enter your code here. Read input from STDIN. Print output to STDOUT
a=int(input())
s_a=set(map(int,raw_input().split()))
b=int(input())
s_b=set(map(int,raw_input().split()))
print(len(s_a.symmetric_difference(s_b)))

# Exercise 31 - Sets - Set .add()
# Enter your code here. Read input from STDIN. Print output to STDOUT

n=int(raw_input())
s = set()
for i in range(n):
    s.add(raw_input())
print(len(s))


# Exercise 32 - Sets - Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, raw_input().split()))
m= int(input())
for i in range(m):
    liste = list(raw_input().split())
    if liste[0]=='pop':
        s.pop()
    elif liste[0]=='remove':
        s.remove(int(liste[1]))
    else:
        s.discard(int(liste[1]))
print(sum(s))


# Exercise 33 - Sets - Set .union() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
s_n = set(map(int, raw_input().split()))
b=int(input())
s_b = set(map(int, raw_input().split()))
print(len(s_n.union(s_b)))


# Exercise 34 - Sets - Set .intersection() Operation
n=int(input())
s_n= set(map(int, raw_input().split()))
b=int(input())
s_b= set(map(int, raw_input().split()))
print(len(s_n.intersection(s_b)))

# Exercise 35 - Sets - Set .difference() Operation
a=int(input())
s_a=set(map(int, raw_input().split()))
b= int(input())
s_b=set(map(int,raw_input().split()))
print(len(s_a.difference(s_b)))


# Exercise 36 - Sets - Set .symmetric_difference() Operation
m,setm=(int(raw_input()),raw_input().split())
n,setn=(int(raw_input()),raw_input().split())
set1=set(setm)
set2=set(setn)
diff1=set1.difference(set2)
diff2=set2.difference(set1)
union=(list((diff1.union(diff2))))
for i in range(len(union)):
    union[i]=int(union[i])
union=sorted(union)
for i in range(len(union)):
    print(union[i])
# Exercise 37 - Sets - Set Mutations
a = int(input())
s1 = set(map(int, raw_input().split()))
N = int(input())

for _ in range(N):
    command = list(raw_input().split())
    s2 = set(map(int, raw_input().split()))
    if(command[0] == "intersection_update"):
        s1.intersection_update(s2)
    elif(command[0] == "update"):
        s1.update(s2)
    elif(command[0] == "symmetric_difference_update"):
        s1.symmetric_difference_update(s2)
    elif(command[0] == "difference_update"):
        s1.difference_update(s2)

print(sum(s1))


# Exercise 38 - Sets - The Captain's Room
k=int(input())
numbers= list(map(int, raw_input().split()))
set_numbers=set(numbers)
print(((sum(set_numbers)*k)-(sum(numbers)))//(k-1))


# Exercise 39 - Sets - Check Subset
n=int(input())
for i in range(n):
    a=int(input())
    set_a=set(map(int, raw_input().split()))
    b=int(input())
    set_b=set(map(int, raw_input().split()))
    if (len(set_a.intersection(set_b)))==a:
        print(True)
    else:
        print(False)
# Exercise 40 - Sets - Check Strict Superset
A=set(map(int, raw_input().split()))
n=int(input())
res=True
for _ in range(n):

    B=set(map(int, raw_input().split()))
    res=res and (len(A.intersection(B))==len(B) and len(A)>len(B))

print(res)

# Exercise 41 - Collections - collections.Counter()
x=int(input())
sizes= list(map(int,raw_input().split()))
n=int(input())
earn=0
for i in range(n):
    y=list(map(int, raw_input().split()))
    if y[0] in sizes:
        earn=earn+y[1]
        sizes.remove(y[0])
print(earn)


# Exercise 42 - Collections - DefaultDict Tutorial
from collections import defaultdict
n,m=map(int,raw_input().split())

d = defaultdict(list)
liste=[]

for i in range(n):
    d[raw_input()].append(i+1) 

for i in range(m):
    liste=liste+([raw_input()])
for i in liste: 
    if i in d:
        print " ".join(map(str,d[i]))
    else:
        print -1


# Exercise 43 - Collections - Collections.namedtuple()
from collections import namedtuple

n = int(input())
columns = raw_input().split()
st = namedtuple('student',columns)
s = 0
for i in range(n):
    
    col1, col2, col3,col4 = raw_input().split()
    student = st(col1,col2,col3,col4)
    s += int(student.MARKS)
print((s/n))

# Exercise 44 - Collections - Collections.OrderedDict()
from collections import OrderedDict

n = int(input())
dic = OrderedDict()
for i in range(n):
    item = raw_input().split(' ')
    price = int(item[-1])
    item_name = " ".join(item[:-1])
    if item_name in dic.keys():
        dic[item_name] += price
    else:
        dic[item_name] = price

for i in range(len(dic.keys())):
    ch=str(dic.keys()[i])+' '+str(dic.values()[i])
    print(ch)


# Exercise 45 - Collections - Word Order

# Exercise 46 - Collections - Collections.deque()
n=int(input())
from collections import deque
d = deque()
for i in range(n):
    command=raw_input().split()
    if command[0]=='append':
        d.append(command[1])
    elif command[0]=='appendleft':
        d.appendleft(command[1])
    elif command[0]=='pop':
        d.pop()
    elif command[0]=='popleft':
        d.popleft()

print(' '.join(d))

# Exercise 47 - Collections - Company Logo


# Exercise 48 - Collections - Piling Up!
T= int(input())
for _ in range(T):
    n=int(input())
    side_len= map(int,raw_input().split())
    mini=min(side_len)
    index=side_len.index(mini)
    if (side_len[:index]==sorted(side_len[:index],reverse=True)) and (side_len[index+1:]==sorted(side_len[index+1:])):
        print('Yes')
    else:
        print('No')

# Exercise 49 - Date time - Calendar Module
import calendar
date=map(int,raw_input().split())
days=['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY']
print(days[calendar.weekday(date[2], date[0], date[1])])


# Exercise 50 - Date time - Time Delta
import math
import os
import random
import re
import sys

# Complete the time_delta function below.
def time_delta(t1, t2):
    from datetime import datetime, timedelta
    
   
    date1=datetime.strptime(t1[0:24],'%a %d %b %Y %H:%M:%S')
    if t1[25]=='+':
        date1-=timedelta(hours=int(t1[26:28]),minutes=int(t1[28:]))
    elif t1[25]=='-':
        date1+=timedelta(hours=int(t1[26:28]),minutes=int(t1[28:]))
    
    date2=datetime.strptime(t2[0:24],'%a %d %b %Y %H:%M:%S')
    if t2[25]=='+':
        date2-=timedelta(hours=int(t2[26:28]),minutes=int(t2[28:]))
    elif t2[25]=='-':
        date2+=timedelta(hours=int(t2[26:28]),minutes=int(t2[28:]))
    return (int(abs((date1-date2).total_seconds())))
if __name__ == '__main__':
    

    n = int(input())

    for _ in range(n):
        t1 = raw_input()
        t2 = raw_input()

        delta = time_delta(t1, t2)

        print(delta)

    

# Exercise 51 - Exceptions -
T = int(input())
for _ in range(T):
        n, m = map(str, raw_input().split())
        try:
                print(int(n) // int(m))
        except ZeroDivisionError as e:
                ch=('Error Code: ' +str(e))
                print(ch)
        except ValueError as e:
                ch=('Error Code: ' + str(e))
                print(ch)


# Exercise 52 - Built-ins - Zipped!
n,x= map(int, raw_input().split() )
marks=[]
for i in range(x):
    marks.append(map(float,raw_input().split()))

Y=zip(*(marks))
for i in range(n):
    print(sum(list(Y[i]))/x)


# Exercise 53 - Built-ins - Athlete Sort
import math
import os
import random
import re
import sys



arr = [list(map(int, raw_input().split())) for _ in range(int(raw_input().split()[0]))]

k = int(input())

print("\n".join([" ".join(map(str, data)) for data in sorted(arr, key = lambda x : x[k])]))



# Exercise 54 - Built-ins - Ginorts
string=raw_input()
lower=[]
upper=[]
odd=[]
even=[]
for i in range(len(string)):
    if string[i].isalpha():

        if string[i].islower():
            lower.append(string[i])
        elif string[i].isupper():
            upper.append(string[i])
    else:
        if int(string[i])%2==0:
            even.append(string[i])
        else:
            odd.append(string[i])
ch=sorted(lower)+sorted(upper)+sorted(odd)+sorted(even)
print("".join(ch))

# Exercise 55 - Map and lambda function
cube = lambda x: x**3 # complete the lambda function 

def fibonacci(n):
    # return a list of fibonacci numbers
    fib=[0,1]
    for i in range(2,n):
        fib.append(fib[i-2] + fib[i-1])
    return(fib[0:n])


# Exercise 56 - Regex - Detect Floating Point Number
T=int(input())
for i in range(T):
    import re
    if (re.search(r'^[-+]?[0-9]*\.[0-9]+$', raw_input()))==None:
        print(False)
    else:
        print(True)
    

# Exercise 57 - Regex - Re.split()
regex_pattern = r"\D+"	# Do not delete 'r'.

# Exercise 58 - Regex - Group(), Groups() & Groupdict()

# Exercise 59 - Regex - Re.findall() & Re.finditer()

# Exercise 60 - Regex - Re.start() & Re.end()

# Exercise 61 - Regex - Regex Substitution

# Exercise 62 - Regex - Validating Roman Numerals
regex_pattern = r"^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"   # Do not delete 'r'. 

# Exercise 63 - Regex - Validating phone numbers
import re
n=int(input())
for i in range(n):
    if re.match(r'[7-8-9]\d{9}$',raw_input())==None:   
        print('NO')  
    else:  
        print ('YES')  


# Exercise 64 - Regex - Validating and Parsing Email Addresses
import re
n = int(input())
for i in range(n):
    email = raw_input()
    if re.match(r'[a-zA-Z][\w\-\.]* <[a-zA-Z][\w\-\.]*@[a-zA-Z]+\.[a-zA-Z]{1,3}>$', email):
        print(email)

# Exercise 65 - Regex - Hex Color Code
import re
n=int(input())
for i in range(n):
    color=raw_input()
    css=color.split()

    if len(css)>1  and  '{' not in css:
        css=re.findall(r'#[a-fA-F0-9]{3,6}',color)
        for i in (css):
            print(i)

# Exercise 66 - Regex - HTML Parser - Part 1

# Exercise 67 - Regex - HTML Parser - Part 2

# Exercise 68 - Regex - Detect HTML Tags, Attributes and Attribute Values

# Exercise 69 - Regex - Validating UID
import re
n=int(input())
for i in range(n):
    liste=raw_input()
    x = "".join(sorted(liste))
    try:
        assert re.search(r'[A-Z]{2}', x)
        assert re.search(r'\d\d\d', x)
        assert not re.search(r'[^a-zA-Z0-9]', x)
        assert not re.search(r'(.)\1', x)
        assert len(x) == 10
    except:
        print('Invalid')
    else:
        print('Valid')


# Exercise 70 - Regex - Validating Credit Card Numbers

# Exercise 71 - Regex - Validating Postal Codes

# Exercise 72 - Regex - Matrix Script

# Exercise 73 - Xml - XML 1 - Find the Score
def get_attr_number(node):
    # your code goes here
    length=len(node.attrib)
    return (length+sum((get_attr_number(x) for x in node)))

# Exercise 74 - Xml - XML 2 - Find the Maximum Depth
maxdepth = 0

def depth(elem, level):
    global maxdepth
    # your code goes here
    depths = [ depth(e,level+1)   for e in elem ]
    if not depths:
        return level + 1
    else:
        x = max(depths)
        if level == -1:
            maxdepth = x
        else:
            return x



# Exercise 75 - Closures and decorators - Standardize Mobile Number Using Decorators

# Exercise 76 - Closures and decorators - Decorators 2 - Name Directory

# Exercise 77 - Numpy - Arrays
def arrays(arr):
    # complete this function
    # use numpy.array
    return(numpy.array(arr[::-1],float))

# Exercise 78 - Numpy - Shape and Reshape

# Exercise 79 - Numpy - Transpose and Flatten
import numpy
n, m = map(int, raw_input().split())

array = numpy.array([raw_input().strip().split() for _ in range(n)],int)
print (array.transpose())
print (array.flatten())

# Exercise 80 - Numpy - Concatenate

# Exercise 81 - Numpy - Zeros and Ones
import numpy
n = tuple(map(int,raw_input().split()))
zeros = numpy.zeros(n, dtype=numpy.int)
ones = numpy.ones(n, dtype=numpy.int)
print (zeros)
print (ones)


# Exercise 82 - Numpy - Eye and Identity
import numpy
n,m = map(int,raw_input().split())
numpy.set_printoptions(sign=' ')
print(numpy.eye(n,m))

# Exercise 83 - Numpy - Array Mathematics

# Exercise 84 - Numpy - Floor, Ceil and Rint
import numpy
numpy.set_printoptions(sign=' ')
A = numpy.array(raw_input().split(),float)
print(numpy.floor(A))
print(numpy.ceil(A))
print(numpy.rint(A))

# Exercise 85 - Numpy - Sum and Prod
import numpy
n, m = map(int, raw_input().split())
A = numpy.array([raw_input().split() for _ in range(n)], int)
print(numpy.product(numpy.sum(A, axis=0)))

# Exercise 86 - Numpy - Min and Max
import numpy
n, m = map(int, raw_input().split())
A = numpy.array([raw_input().split() for i in range(n)], int)
print(numpy.max(numpy.min(A, axis=1)))


# Exercise 87 - Numpy - Mean, Var, and Std

# Exercise 88 - Numpy - Dot and Cross

# Exercise 89 - Numpy - Inner and Outer

# Exercise 90 - Numpy - Polynomials

# Exercise 91 - Numpy - Linear Algebra



# ===== PROBLEM2 =====



# Exercise 92 - Challenges - Birthday Cake Candles

# Exercise 93 - Challenges - Kangaroo

# Exercise 94 - Challenges - Viral Advertising

# Exercise 95 - Challenges - Recursive Digit Sum

# Exercise 96 - Challenges - Insertion Sort - Part 1

# Exercise 97 - Challenges - Insertion Sort - Part 2      



    

