"""
consider a csv file as follows:
name;Cols;Rows;Threads;kernelTime;TotalTime

For such file consider all the files with the same Threads ans image and make an avg of the Kernel and total time"""
import csv
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def parse_csv(input_file):
    data = {}
    with open(input_file, mode='r') as file:
        reader = csv.reader(file, delimiter=';')
        header = next(reader)  # Skip the header
        for row in reader:
            if len(row) < 6:
                continue  # Skip rows that do not have enough data
            name, cols, rows, threads, kernel_time, total_time = row
            key = (threads, cols, rows)
            if key not in data:
                data[key] = {'kernel_times': [], 'total_times': [], 'names': []}
            data[key]['kernel_times'].append(float(kernel_time))
            data[key]['total_times'].append(float(total_time))
            data[key]['names'].append(name)
    return data

def write_avg_to_csv(data):
    avgtimes = []
    for key, values in data.items():
        avg = {}
        threads, cols, rows = key
        avg_kernel_time = sum(values['kernel_times']) / len(values['kernel_times'])
        avg_total_time = sum(values['total_times']) / len(values['total_times'])
        avg["name"] = values['names'][0] 
        avg["kernel_time"] = avg_kernel_time
        avg["total_time"] = avg_total_time
        avg["threads"] = threads

def timeGraph(data):
    
