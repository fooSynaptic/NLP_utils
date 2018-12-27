'''

Single case is one of the programing style, we employed single case to hadle classes with only one instance.

For different position in code we can reach the only one object instance with the gurantee of Single case mode.
In the case the instance not exists, the pattern will build a new instance, if exists, it will return directly.
Single case is one kind of class.
'''

#使用函数装饰器实现单例
def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner
    
@singleton
class Cls(object):
    def __init__(self):
        pass

cls1 = Cls()
cls2 = Cls()
print(id(cls1) == id(cls2))


# 使用类装饰器实现单例

class Singleton(object):
	def __init__(self, cls):
		self._cls = cls
		self._instance = {}

	def __call__(self):
		if self._cls not in self._instance:
			self._instance[self._cls] = self._cls()
		return self._instance[self._cls]

@Singleton
class Cls2(object):
	def __init__(self):
		pass

cls1 = Cls2()
cls2 = Cls2()
print(id(cls1) == id(cls2))


#使用new关键字实现单例模式

class Single(object):
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
    def __init__(self):
        pass

single1 = Single()
single2 = Single()
print(id(single1) == id(single2))


#使用metaclass实现单例模式

#first，了解使用type创造类的方法
def func(self):
    print("do sth")

Klass = type("Klass", (), {"func": func})

c = Klass()
c.func()


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Cls4(metaclass=Singleton):
    pass

cls1 = Cls4()
cls2 = Cls4()
print(id(cls1) == id(cls2))
