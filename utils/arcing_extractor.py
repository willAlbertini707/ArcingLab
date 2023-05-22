'''
This module acts as a data pipline for Cal Poly AERO 356-03 Lab 2
arcing experiment

'''

# external imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Any, Optional
import os, re

# internal imports
from .data_extractor import ExtractorSkeleton


class ArcingDataExtractor(ExtractorSkeleton):

	'''
	for arcing extractor class, data needs to be in csv or txt with comma
	delimiters and the following column names:

	columns = [pressure, voltage, current]

	'''

	# capactance for panel
	C = 2e-6 # F

	# column names for table
	column_names = ['pressure', 'voltage', 'current']

	def __inti__(self):
		self._frame_dict = None


	def fit_extract(self, directory: str) -> Any:
		'''
		Uses given directory to extract all data and store it
		in a dictionary of tests and dataframes

		'''

		# create dataframe dictionary
		self._frame_dict = {}

		# loop through directory
		for file in os.listdir(directory):

			# check for relevant files
			if file[-3:] == 'csv' or file[-3:] == 'txt':

				# save the name of the test
				test_name = file[:-4]

				# create new path to data
				path = os.path.join(directory, file)

				# create temporary data frame
				temp_df = pd.read_csv(path)

				# check if column names follow format
				if sorted(temp_df.columns.to_list()) != sorted(self.column_names):

					raise Exception(f'Column names must be {self.column_names}')

				# pull distance measurement from file name
				if '2D' in file:
					length = 1.0
				else:
					length = 0.5

				# create pressure distance column
				temp_df['torr_in'] = temp_df.pressure * length
				
				# add frame to dictionary
				self._frame_dict[test_name] = temp_df.copy()

		# check to make sure dictionary was added to
		if not self._frame_dict:
			raise Exception("No csv of txt files detected")

		return self


	def label_plot(self, xlabel: str, ylabel: str, title: str, xlim: str, ylim: str) -> None:
		'''
		label plots

		'''
		# plt.ylim(ylim)
		plt.xlim(xlim)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.legend()


	def color_generator(self, n: int) -> np.ndarray:
		'''
		color generator to give each value a unique color 
		(unless more than n dataframes are provided)
		'''
		colors = cm.rainbow(np.linspace(0, 1, n))
		i = 0
		while True:
			if i < len(colors):
				yield colors[i]

			else:
				i = 0
				yield colors[i]

			i += 1


	def plot_results(self, savefig: bool = False, x: str ='torr_in', y: str='voltage', 
		labels: List[str] = ['Pressure Distance (torr inch)', 'Voltage (kV)'], 
		title: str = 'Arcing Voltage vs Pressure Distance', 
		hspace: float = 0.4, wspace: float = 0.4, alpha: float = 0.2, 
		xlim: Optional[List[float]] = None) -> None:

		# check for valid dataframe
		if not self._frame_dict:
			raise Exception("Run fit_extract first with appropriate arguments")

		# create color dictionary
		color_dict = self.color_generator(len(self._frame_dict))

		# pre-allocate lists for mins and maxes
		x_min, x_max = [], []
		y_min, y_max = [], []

		# loop through frame and plot curves
		for test, df in self._frame_dict.items():

			# keep track of min and max values
			x_min.append(df[x].min())
			x_max.append(df[x].max())
			y_min.append(df[y].min())
			y_max.append(df[y].max())

			# get color from dictionary
			color = next(color_dict)

			# plot the data
			plt.loglog(df[x], df[y], color = color, alpha = alpha, label = f"{test}")

		if not xlim:
			xlim = [max(x_min), max(x_max)*1.1]

		ylim = [0, max(y_max)*1.2]

		self.label_plot(*labels, title, xlim, ylim)

		if savefig:
			plt.savefig(f"{title}.png")

		plt.show()


	@property
	def frame_dict(self) -> Dict[str, pd.DataFrame]:
		# frame_dict getter
		return self._frame_dict.copy()