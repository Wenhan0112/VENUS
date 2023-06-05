"""
A class representing the union of intervals.  
    Date: May 2023
    Author: Wenhan Sun
"""

import pandas as pd

class Intervals():
    """
    A class of unions of intervals. 
    @field intervals (list[pandas.Interval]): The list of intervals. 
    """
    def __init__(self, intervals: list[pd.Interval]):
        """
        Constructor.
        @params intervals (list[pandas.Interval]): The list of intervals. 
        """
        self.intervals = intervals.copy()

    def add(self, i: pd.Interval):
        """
        Add an additional interval. 
        @params i (pandas.Interval): The additional interval added.
        """
        self.intervals.append(i)

    def isin(self, el: float) -> bool:
        """
        Check if an element is in the union of intervals. 
        @params el (float): The element.
        @return (bool): True iffthe element is in the union of intervals. 
        """
        for interval in self.intervals:
            if el in interval:
                return True
        return False
