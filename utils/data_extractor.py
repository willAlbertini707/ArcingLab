# external imports
from abc import ABC, abstractmethod


'''
This module acts as a blue print for the two data extracting
classes to be used for data analysis/processing

'''


class ExtractorSkeleton(ABC):
	'''
	base class for data extractor

	'''

	@abstractmethod
	def fit_extract(self):
		'''
		Fits model to given directory and turns all csv/txts into dataframes

		'''
		pass


	@abstractmethod
	def plot_results(self):
		'''
		visually display the data

		'''

		pass







