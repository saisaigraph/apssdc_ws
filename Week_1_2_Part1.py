
#Basic Arithmetic

1+1

# Sub
2-1

2*2

# Div
3/2

# Division always returns floats!
1/1

2 ** 3

2 ** (1/2)

# Order of Operations
1 + 2 * 1000 + 1

(1 + 2) * (1000+1)

# Integer Arithmetic
14/4


14//4


14%4

# ************************************************************

'Mera Bharath Mahaan'

"Mera Bharath Mahaan"

# Comment. 

"Mera Bharath Mahaan" # This is creating a text cup

# Single or double quotes are okay.
Mera Bharath Mahaan


#Basic Printing of Strings

'Namasthe'

'Mera Bharath Mahaan'

'Namasthe'
'Mera Bharath Mahaan'





print('Namasthe')
print('Mera Bharath Mahaan')




'
She said, "Thank you! It's mine."
She said, "Thank you! It's mine."
She said, "Thank you! It's mine."
'



"
She said, "Thank you! It's mine."
She said, "Thank you! It's mine."
She said, "Thank you! It's mine."
"







'''
She said, "Thank you! It's mine."
She said, "Thank you! It's mine."
She said, "Thank you! It's mine."
'''


print('''
"Namasthe"
'Mera Bharath Mahaan'
"Hamara Bharath Mahaan"
"Namasthe"
'Mera Bharath Mahaan'
"Hamara Bharath Mahaan"
"Namasthe"
'Mera Bharath Mahaan'
"Hamara Bharath Mahaan"
''')

#*** 

radius = 3

type(radius)

pi = 3.14

type(pi)

area_of_circle = pi * (radius**2)
area_of_circle

type(area_of_circle)



type('dog')

type('1')

type(1)

mera = 'Mera Bharath'

mahaan = ' Mahaan'

id(mera)

'Nescafe ' + 'Coffee'


'Nescafe ' + 'Nescafe '  + 'Nescafe ' 



10*'Nescafe ' 



3*'Nescafe '  + 'Coffee'



7+2


'7'+2

'7'+'2'

'7' + str(2)

str(2)

int('2')

float(2)

int(2.0)

str(2.0)



















nescafe = coffee

nescafe = 'coffee'

nescafe


person = 'Sai'
person

coffee = input('What is your favorite coffee: ')

coffee


'''input and print.'''
applicant = input("Enter the applicant's name: ")

interviewer = input("Enter the interviewer's name: ")


# Input and Output 
person = input('Enter your name: ')
print('Hello', person, '!')



age = int(input('Please enter your age: '))
age

if age < 21 :
	print('You are not eligible for Driving License.')

#******************** ********************
# *** Lists & Dicts 

# Lists

a = 1

type(a)

b = 'Mera Bharath Mahaan'

type(b)

my_list = [1,2,3]

my_list

my_list2 = ['Nescafe','Bru','Sunrsie']

my_list2

my_list3 = [1,2,3,'Nescafe','Bru','Sunrise']
my_list3

a = 100
b = 200
c = 300

my_list4 = [a,b,c]
my_list4

# Indexing and Slicing
# same as in a string!

mylist = ['nescafe','bru','cappaccino','mocha']
mylist

mylist[0]

mylist[0:3]

mylist[-1]

mylist[-2]

mylist[2:]

mylist[:2]

#The len function

len('Mera Bharath')

len(100)

len('100')

len(mylist)

# Create an empty list
empty_list = []

# Create a list of numbers
integer_list = [2, 3, 5]


















































#******************** ********************
'''
# ************************************************************ E13
'''



# *** Numpy ***

_list = [0,1,2,3,4]

_arr1d = np.array(_list)

import numpy as np

_arr1d = np.array(_list)

_arr1d

import numpy as anjum

_arr1d = anjum.array(_list)
_arr1d


print(type(_list))

print(type(_arr1d))

_list + 2 
#err

_arr1d + 2

print(_arr1d * 4)

print(_arr1d / 2)

print(_arr1d - 2)

#***

np.zeros(5)

np.ones(5)

np.twos(5)

np.ones(5) * 2

np.ones(5) * 3


np.zeros((2,2))

np.ones((2,2))

np.ones((2,2)) * 2

np.ones((2,3))


#***


# Merging arrs
arr1 = np.matrix([1,2,3])
print(arr1)

arr2 = np.matrix([4,5,6])
print(arr2)

print('Horizontal Append:', np.hstack((arr1, arr2)))

print('Vertical Append:', np.vstack((arr1, arr2)))

#***

arr2d1 = np.array([[1,2],[3,4]])
print(arr2d1)

arr2d2 = np.array([[5,6],[7,8]])
print(arr2d2)

print('Horizontal Append:', np.hstack((arr2d1, arr2d2)))

print('Vertical Append:', np.vstack((arr2d1, arr2d2)))




# ************************************************************


x = np.array(range(100))

x

x.ndim 
# 1, the number of axes (dimensions).

x.shape 
# (100,)

x.size 
# 100

x.dtype 
# dtype('int32')

y = np.array([[1.0,2,3], [4,5,6], [7,8,9]])

y

y.ndim 
# 2

y.shape 
# (3, 3)

y.size 
# 9

y.dtype 
# dtype('float64') since we put 1.0 as an element

np.full((5,5), 53)



x = np.array([[1, 2, 3], [4, 5, 6]])
x

np.full(x.shape, 53)


np.eye(5)

np.eye(5,3)



x = np.array(range(90))
x

x = x.reshape((10,9))
x

x.reshape((3,3,10))

x = np.array(range(90)).reshape(10,3,3)
x

x.ravel()

x = np.array(range(90)).reshape(10,3,3)
x

x.T

x = np.array(range(1, 101))
x

# Math ops
x.sum() 
# 5050

x.min() 
# 1

x.max() 
# 100

x.mean() 
# 50.5

x.std() 
# 28.86607004772212


np.median(x)

np.sqrt(x)

x = np.array(range(100)).reshape((10, 10))

x

# Element Selection
x[0, 1]

x[2, 9]

# Subarray Selection
x[1]


x[:, 2]


x[1:3]


x[:, 1:3]


x[1:3, 1:3]


x[[0,3,9]]
# get particular rows


x[[0,3,9], 2]


x[[0,3,9], [0,9,3]]


#Boolean Indexing
bool_index = (x > 50)
bool_index

x[bool_index]


# Value Changes
x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
x


x[0, 0, 0] = 100

x


x[:, 1, 1] = 99 
# indices: [0,1,1], [1,1,1], [2,1,1]
x
#all the elements with 2nd and 3rd dimensions with index position 1



x[0, :, :] = -888 # indices [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2]
x





































#******************** ********************
# ****
# ***********************************
# Tuples & Sets
# ***********************************


lst = [1,2,3]
lst

lst[0] = 'Nescafe'

lst

# Tuples
# Can create a tuple with mixed types
t = (1,2,3)
t

# Check len just like a list
len(t)

# Can also mix object types
t = ('Coffee',2)
t

t[0]

t[0] = 'Tea'

type(t)

t = (1, 4, 5)
t

tuple0 = 1, 4, 5
print(tuple0)


tuple1 = (1, 2, 'coffee')
print(tuple1)


#tuple2 = (4, 7, ('a', 'b'), lambda x: x+1)
#print(tuple2)


tuple3 = () 
print(tuple3)

type(tuple3)

tuple4 = 'coffee'
tuple4

type(tuple4)

# Create a tuple using tuple()
tuple5 = tuple(['a', 'b'])
print(tuple5)


tuple6 = 'Nescafe'
print(tuple6)

type(tuple6)

tuple6 = 'Nescafe',
print(tuple6)



tuple6 = ('Nescafe')
print(tuple6)


tuple6 = ('Nescafe',)
print(tuple6)




# Dict
# []
# ()
# {}
trayPens = {'cap1': 'Reynolds', 'cap2': 'Parker'}
trayPens

tuple7 = tuple({'cap1': 'Reynolds', 'cap2': 'Parker'})
print(tuple7)

{'a': 1, True: 4}


tuple7 = tuple({'a': 1, True: 4})
print(tuple7)

#***


#***


tray1 = ['Nescafe','Nescafe','Taj','Brook Bond']
tray1 

tray2 = set(['Nescafe','Nescafe','Taj','Brook Bond'])
tray2 

tray2 = set(tray1)
tray2 


s1 = set('avinash')
s1









#*** Kw1


a = 1

b = 2

1 = 1

and = 1

or = 1















	























'''
# ************************************************************ E12
'''

# *** Dicts 

tray = ['Nescafe', 'Bru', 'Sunrise']
tray

tray[0]

# Dicts 

cTray = {'Coffee': 'Nescafe', 'Filter':'Bru', 'Instant':'Sunrise'}
cTray

cTray['Coffee']


wtr1 = ['Coffee', 'Tea', 'Green Tea']

wtr1[0]

# Dicts 
wtr2 = {'Nescafe': 'Coffee', 'Bru':'Filter', 'Sunrise':'Instant'}
wtr2


wtr2['Nescafe']

wtr2['Bru']

wtr2['Bru'] = 'Filter Coffee'

wtr2

wtr2['Sunrise'] = 'Frappes'

wtr2

wtr2['Continental'] = 'Latte'

wtr2

wtr2['expresso'] = 'Latte'
wtr2

wtr2['expresso'] = 'Mocha'
wtr2


wtr2['expresso2'] = 'Mocha'
wtr2

wtr2.keys()

wtr2.values()







# Indexing and Reverse Indexing
# Create a list for indexing
number_list = [0, 1, 2, 3, 4, 5, 6]

# The first element
number_list[0]


number_list[2]


number_list[-1]


number_list[-2]



# Slicing
# The original list
long_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

long_list[5:]


long_list[:5]



# The 3rd to the 9th
long_list[2:9]

long_list[::3]


long_list[3:-2]






#***

# Iteration
# Create a list for iteration
tray = ["Nescafe", "Bru", "Sunrise"]

# Regular iteration
for cofee in tray:


# Regular iteration
for cofee in tray:
     # do something with each of the pets
     pass
 



tray = ["Nescafe", "Bru", "Sunrise"]
for coffee in tray:
     print(coffee)

for coffee in reversed(tray):
     print(coffee)





#***

import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

x = np.linspace(0, 5, 11)
x

x = np.linspace(0, 5, 3)
x

x = np.linspace(0, 5, 11)
x

y = x ** 2
y

# Functional
plt.plot(x, y) 
# 'r' is the color red
# In Jupyter


%matplotlib inline


plt.plot(x, y, 'r') 
# 'r' is the color red



plt.plot(x, y, 'r--') 
# 'r' is the color red



# In others
plt.plot(x, y, 'r') # 'r' is the color red
plt.show()



plt.plot(x, y, 'r') # 'r' is the color red
plt.xlabel('X Axis Title Here')



plt.plot(x, y, 'r') # 'r' is the color red
plt.xlabel('X Axis Title Here')
plt.ylabel('Y Axis Title Here')



plt.plot(x, y, 'r') # 'r' is the color red
plt.xlabel('X Axis Title Here')
plt.ylabel('Y Axis Title Here')
plt.title('String Title Here')
#plt.show()




plt.plot(x, y, 'r') # 'r' is the color red
plt.xlabel('X Axis Title Here')
plt.ylabel('Y Axis Title Here')
plt.title('String Title Here')
plt.show()






#***


#*** 

# Reverse Lists
# Declare a list for reversing
names = ['Nescafe', 'Bru', 'Sunrise', 'Cappuccino']
names

# Using the list's reverse() method
names.reverse()
names


# Using the reversed() method, which returns an iterable
names_reversed = list(reversed(names))
names_reversed


# Using the slicing-based method
names_reversed1 = names_reversed[::-1]
names_reversed1


names_reversed


# Check Element’s Presence
# Declare a list for checking
integers = [2, 3, 4, 5, 8]
integers

# Check if 8 is in the list
8 in integers
True

tray

'Nescafe' in tray

'Nescafe' not in tray




#***



# Adv unpacking
tray = ['Gift1', 'Nescafe', 'Bru', 'Sunrise', 'Gift5']
tray

# when we're only interested in the first and last element
a, *_, b = tray

a, *remaining, b = tray

# elements as a list and two last elements
*c, d, e = tray




#***
















# ***
# ************************************************************


# ****** 06 ****** ******
# Moduless & Pkgs
# ****** ****** ******


print "a" 
# is a func

os.listdir() 
#v get an err

import os 
# os is a module

os.listdir() 

dir(os)

os?

os.__file__

C:\Users\admin\AppData\Local\Continuum\Anaconda2\Lib

os?

os??

help(os)

help()

quit


sqrt(4)

math.sqrt(4)

import math 
# is a module

math.sqrt(4)

sqrt(4)

from math import sqrt

sqrt(4)

dir(math)


#ignore
#conda install flask #Anaconda comes with flask
#pip install python-pptx #if Anaconda does nt cm with a mod



























'''
# ************ o PB1
'''

# **********************************
# Funcs
#**** Functions ***

Functions
1. Functions Basics

print("Mera Bharath Mahaan!") 
















print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 
print("Mera Bharath Mahaan!") 




def greet(): 
    print("Mera Bharath Mahaan!") 





def greet()
print("Mera Bharath Mahaan!") 





def greet(): 
    print("Mera Bharath Mahaan!") 

greet() 





greet()
greet()
greet()
greet()
greet()
greet()
greet()
greet()
greet()
greet()
greet()
greet()
greet()
greet()
greet()
greet()
greet()
greet()





def greetN(n): 
    i = 0 
    while(i < n ): 
        print("Mera Bharath Mahaan!") 
        i = i + 1 

greetN(5) 


def greetN(n): 
    '''Prints "Mera Bharath Mahaan" n times" ''' 
    i = 0 
    while(i < n ): 
        print("Hamara Bharath Mahaan!") 
        i = i + 1 





print("Hamara Bharath Mahaan!") 
def greet(): 
    print("Mera Bharath Mahaan!") 
print("Hamara Bharath Mahaan!") 
















# Local vs Global
nescafe = 0
nescafe

def ravi_house(brookbond, taj):
	lipton = 0
	lipton += 1
	brookbond = taj + lipton
	print(brookbond)

nescafe

ravi_house(1, 2)

brookbond

taj

bru = nescafe + 1
bru

bru = brookbond + 1

print(nescafe)

def harika():
	global nescafe
	nescafe = 'Latte'
	print(nescafe)

harika()


def harika():
	print(brookbond)

harika()


nescafe

def ravi():
	nescafe = 'Expresso'
	print(nescafe)

ravi()

harika()


print(bru)





# Adv Functs & Classes

# *******  ****22*** *******
# *****************************
# Adv Funcs
# *****************************

2. Parameter Passing

def sumNums( n1, n2, n3 ): 
    sum = n1 + n2 + n3 
    return sum 

tot = sumNums( 10, 20, 30) 
print("Total = ", tot )     


tot = sumNums( 10, 50  ) 











    












    

def sumNums( n1, n2=20, n3=30 ): 
    sum = n1 + n2 + n3 
    return sum 

tot = sumNums( 10  ) 
print("Total = ", tot ) 











    

tot = sumNums( 10, 40 ) 
print("Total = ", tot ) 











    

tot = sumNums( 100, 200, 300 ) 
print("Total = ", tot ) 









def sumNums( n1,n2,n3): 
    sum = n1+n2+n3
    return sum 

tot = sumNums(n3=10,n2=30,n1=50) 
print("Total = ", tot ) 

tot = sumNums(n3=10,30,n2=50) 

















'''
# ************************************************************ E16
'''

def f1(taj): 
    brookbond = 0 
    brookbond = brookbond + taj    
    return brookbond 

f1(10)


def f2(taj,waghbakri): 
    brookbond = 0 
    brookbond = brookbond + taj + waghbakri    
    return brookbond 

f2(1,2)



def f3(taj, waghbakri, parivar): 
    brookbond = 0 
    brookbond = brookbond + taj + waghbakri + parivar
    return brookbond 

f3(1,2,3)

...

def f100(taj, waghbarkri, parivar,....tea100)
    brookbond = 0 
    brookbond = brookbond + taj + waghbakri + parivar + ...100
    return brookbond 
f100(1,2,3...100)







def f( *args ): 
    tea = 0 
    for cup in args: 
        tea = tea + cup    
    return tea 

tot = f( 10) 
print("Total = ", tot ) 

tot = f( 10, 20) 
print("Total = ", tot ) 

tot = f( 10, 20, 30,40,50) 
print("Total = ", tot ) 



tot = f( 10, 20, 30, 40, 50 ) 
print("Total = ", tot ) 



# ************ 26











#***






def fruitBasket( **kwargs ): 
    count = 0 
    for key, val in kwargs.items(): 
        printString = "{:20}{:10}".format(key,val) 
        print( printString ) 
        count = count + val  
    return count 

tot = fruitBasket( apples=100, banana=144, pears=77, grapes=200, mangoes=35 ) 
print("-" * 30 ) 
print( "{:20}{:10}".format("Total Fruits: ", tot) ) 


def fruitBasket( **kwargs ): 
    count = 0 
    for key, val in kwargs.items(): 
        printString = "{:20}{:10}".format(key,val) 
        print( printString ) 
        count = count + val  
    return count 

tot = fruitBasket( apples=100, banana=144, pears=77, grapes=200, mangoes=35 ) 
print("-" * 30 ) 
print( "{:20}{:10}".format("Total Fruits: ", tot) ) 





#***








def myFunc( n1, n2, n3, *args, **kwargs ): 
    sum1 = sum2 = sum3 = 0 
    sum1 = n1 + n2 + n3 
    for n in args:  
        sum2 = sum2 + n 
    for k,v in kwargs.items(): 
        sum3 = sum3 + v 
         
    sums = [ sum1, sum2, sum3 ] 
    return sums 

sums = myFunc( 10, 20, 30, 11, 22, 33, 44, one=100, two=200, three=300 ) 
print ( sums ) 

sums = myFunc( 10, 20, 30, one=100, two=200, three=300 ) 
print ( sums ) 







'''
# ************************************************************ 
'''


'''
# ************************************************************ 10c
'''

class Person()

class Person():
	pass

sachin = Person()




class Person():
    def __init__(self):
        print('I m in Constructor')

sachin = Person()



class Person():
    def __init__(self,aName,anAge):
        print('I m in Constructor')
        self.name = aName
        self.age = anAge

sachin = Person()


sachin = Person('Sachin',20)

dhoni = Person('Dhoni',21)


class CricketPlayer():
	def __init__(self,aName,aBat,aBall):
		print('I m in Constr...')
		self.name = aName
		self.bat = aBat
		self.ball = aBall

sachin = CricketPlayer('Sachin','MRF','Guru')	


class BadmintonPlayer():
	def __init__(self,aName,aRacket,aShuttle):
		print('I m in Constr...')
		self.name = aName
		self.racket = aRacket
		self.shuttle = aShuttle

sindhu = BadmintonPlayer('Sindhu','Yonex','Yonex')	



class Person():
    def __init__(self,aName,anAge,aSal):
        print('I m in Constructor')
        self.name = aName
        self.age = anAge
        self.salary = aSal
    def eat(self):
        print('Idly')
    def play(self):
        print('Badminton')
        print('or Cricket')
    def exercise(self):
        print('Jogging')


p1 = Person('Sai')

p1.

p1.name


class CricketPlayer():
	def __init__(self,aName,aBat,aBall):
		print('I m in Constr...')
		self.name = aName
		self.bat = aBat
		self.ball = aBall
	def doBatting(self):
		print('I m ready to Bat')
	def doBowling(self):
		print('I m ready to Ball')

sachin = CricketPlayer('Sachin','MRF','Guru')	

sachin.doBatting()


class BadmintonPlayer():
	def __init__(self,aName,anAge,aRaquet,aShuttle):
		print('I m a Badminton Player')
		self.name = aName
		self.age = anAge
		self.raquet = aRaquet
		self.shuttle = aShuttle
	def doService():
		print('Service')
	def doSmashing():
		print('Smashing')

sindhu = BadmintonPlayer('sindhu',18,'Yonex','Yonex')



class BadmintonPlayer():
	def __init__(self,aName,anAge,aRaquet,aShuttle):
		print('I m a Badminton Player')
		self.name = aName
		self.age = anAge
		self.raquet = aRaquet
		self.shuttle = aShuttle
	def doService(self):
		print('Service')
	def doSmashing(self):
		print('Smashing')

sindhu = BadmintonPlayer('sindhu',18,'Yonex','Yonex')



class BadmintonPlayer():
	def __init__(self,aName,anAge,aRaquet,aShuttle):
		print('I m a Badminton Player')
		self.name = aName
		self.age = anAge
		self.raquet = aRaquet
		self.shuttle = aShuttle
		self.bat = ''
		self.ball = ''
	def doService(self):
		print('Service')
	def doSmashing(self):
		print('Smashing')

sindhu = BadmintonPlayer('sindhu',18,'Yonex','Yonex')


class Car:
	def __init__(self,aName,anAge,brks,tyrs):
		print('I m a Car')
		self.name = aName
		self.age = anAge
		self.brakes = brks
		self.tyres = tyrs
	def start(self):
		print('Started the Car')
	def stop(self):
		print('Stopping')
	def accelerate(self):
		print('Increasing the speed')

honda = Car('Honda', 1, 'MRF', 'CEAT')

''' ******** E o PB2 ********* '''


#*** 

class CricketPlayer():
	def __init__(self,anAge,aName):
		self.name = aName
		self.age = anAge
		self.awards = list()
		self.jerseys = ('White','Blue')
		self.jerseyNums = {'White': 10,'Blue':10}


sachin = CricketPlayer('46', 'Sachin')


class CricketPlayer():
	def __init__(self,anAge,aName):
		self.name = aName
		self.age = anAge
		self.awards = list()
		self.jerseys = ('White','Blue')
		self.jerseyNums = {'White': 10,'Blue':10}
		self.records = set(['Highest Score','Num of Matches','Highest Score'])
		self.coffCups = {'Nescafe','Bru','Nescafe'}

sachin = CricketPlayer('Sachin','46')
