'''
This module acts as a data pipline for Cal Poly AERO 356-03 Lab 2
solar panel arcing experiment

'''

# external imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from typing import Dict, List, Any

# internal imports
from .data_extractor import ExtractorSkeleton


class PanelDataExtractor(ExtractorSkeleton):
	'''
	for panel extractor class, data needs to be in csv or txt with comma
	delimiters and the following column names:

	columns = [current_pre, voltage_pre, current_post, voltage_post]

	'''

	# define constants for code

	# saturation current
	I0 = 1.95e-12  # A/m2 

	# boltzman constant
	k = 1.380649e-23 # m2 kg s-2 K-1

	# elementary charge
	q = 1.60217663e-19 # coulombs

	# area of solar cell
	A = 0.0024 # m2

	# column names
	column_names = ['current_pre', 'voltage_pre', 'current_post', 'voltage_post']


	def __init__(self):
		self._frame_dict = None


	def fit_extract(self, directory: str) -> Any:

		# create file/directory dictionary
		self._frame_dict = {}

		# loop through directory and pull relevant files
		for file in os.listdir(directory):

			# check if file is a csv before adding
			if file[-3:] == 'csv' or file[:-3] == 'txt':

				# grab the name of the test
				name = file[:-4]
				
				# creat file path
				path = os.path.join(directory, file)

				# store the name and dataframe
				temp_df = pd.read_csv(path)

				# check to see if columns are formatted correctly
				if temp_df.columns.to_list().sort() != self.column_names.sort():
					raise Exception(f'Column names must be {self.column_names}')


				# add dataframe to dicitonary
				self._frame_dict[name] = temp_df.copy()

		# check to see if any files were added
		if not self._frame_dict:
			raise Exception("No csv of txt files detected")

		
		return self


	def _current_adjust(self, T: float, I: float, V: float) -> float:

		'''
		adjust the current for a given temperature

		'''
		# calculate corrected current
		I_correct = I - self.I0 * (np.exp(self.q*V/self.k/T) - 1)

		return I_correct


	def apply_current_adjust(self, T: float) -> Any:
		'''
		apply current adjust to all frames

		T: temperature correction in Kelvin
		'''

		# loop through dataframe dictionary and change current values
		if self._frame_dict:

			for key, val in self._frame_dict.items():

				val.current_pre = self._current_adjust(T, val.current_pre, val.voltage_pre)
				val.current_post = self._current_adjust(T, val.current_post, val.voltage_post)
		else:
			raise Exception("Run fit_extract first with appropriate arguments")


		return self


	def add_power(self) -> Any:
		'''
		Add power column (pre and post) to existing tables

		'''

		# add power column pre and post for analysis

		for key, val in self._frame_dict.items():
			val['power_pre'] = val.current_pre * val.voltage_pre
			val['power_post'] = val.current_post * val.voltage_post


		return self


	def label_plot(self, xlabel: str, ylabel: str, title: str, xlim: str, ylim: str, ax: plt.Axes) -> None:
		'''
		label plots

		'''
		ax.set_ylim(ylim)
		ax.set_xlim(xlim)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_title(title)
		ax.legend()


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

 
	def plot_results(self, savefig: bool = False, x: str ='voltage', y: str='current', labels: List[str] = ['Voltage (V)', 'Current (A)'],
		plot_type: str = 'IV', hspace: float = 0.4, wspace: float = 0.4, alpha: float = 0.2) -> None:
		'''
		plots IV relationship

		'''
		# check for valid dataframe
		if not self._frame_dict:
			raise Exception("Run fit_extract first with appropriate arguments")


		# create columns
		x_pre, x_post = x + '_pre', x + '_post'
		y_pre, y_post = y + '_pre', y + '_post'

		# check for valid column names
		for col in [x_pre, x_post, y_pre, y_post]:

			for key, val in self._frame_dict.items():
				if col not in val.columns:
					raise Exception(f"{col} column not found in dataframe(s)")

		x_min, x_max = [], []
		y_min, y_max = [], []

		fig, axes = plt.subplot_mosaic([['a', 'a'],['b','c']], figsize=(10,8))	
		ax = [x[1] for x in axes.items()]

		# use color generator for unique colors
		color_gen = self.color_generator(len(self._frame_dict) + 1)


		for name, df in self._frame_dict.items():

			# keep track of min and max values
			x_min += [df[x_pre].min(), df[x_post].min()]
			x_max += [df[x_pre].max(), df[x_post].max()]
			y_min += [df[y_pre].min(), df[y_post].min()]
			y_max += [df[y_pre].max(), df[y_post].max()]

			line_color = next(color_gen)
			# plot all values on main subplot
			ax[0].plot(df[x_pre], df[y_pre], label=f'{name} pre', color = line_color, alpha = alpha)
			ax[0].plot(df[x_post], df[y_post], linestyle = '--', label =f'{name} post', color = line_color, alpha = alpha)
			
			# plot all pre values on smaller plot
			ax[1].plot(df[x_pre], df[y_pre], label=f'{name}', color=line_color, alpha=alpha)

			# plot all post values on seperate smaller plot
			ax[2].plot(df[x_post], df[y_post], label=f'{name}', color=line_color, alpha=alpha)


		# label plots
		xlim = [max(x_min), max(x_max)*1.1]
		ylim = [0, max(y_max)*1.2]

		self.label_plot(*labels, f'{plot_type.title()} Plot Pre and Post Arcing', xlim, ylim, ax=ax[0])
		self.label_plot(*labels, f'{plot_type.title()} Plot Pre Arcing', xlim, ylim, ax=ax[1])
		self.label_plot(*labels, f'{plot_type.title()} Plot Post Arcing', xlim, ylim, ax=ax[2])

		# space out subplots
		fig.subplots_adjust(hspace=hspace, wspace=wspace)

		plt.plot()

		if savefig:
			fig.savefig('IV_plot.png')


	@property
	def frame_dict(self) -> Dict[str, pd.DataFrame]:
		# frame_dict getter
		return self._frame_dict.copy()