import tkinter as tk
import traceback

class Catcher: 
    def __init__(self, func, subst, widget):
        self.func = func 
        self.subst = subst
        self.widget = widget
    def __call__(self, *args):
        try:
            if self.subst:
                args = apply(self.subst, args)
            return apply(self.func, args)
        except SystemExit:
            raise SystemExit
        except:
            traceback.print_exc(file=open('test.log', 'a'))