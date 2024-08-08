import pandas as pd
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import scipy.io
import numpy as np
import statistics

class MatlabFileReader:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.filename = ""
        self.source = ""
        self.fignbr=""
        self.x_label= "X_Axis"
        self.y_label= "Y_Axis"
        self.graph = [] 

    def read_matfile(self, file_name, source_name="", fig_nbr=""):
        try:
            self.resetLabels()
            # Load the MATLAB file using scipy.io.loadmat
            self.filename = file_name
            file_path = os.path.join(self.directory_path, file_name)
            mat_data = scipy.io.loadmat(file_path, struct_as_record=True)
            self.source = source_name
            self.fignbr = fig_nbr
            #print(mat_data)
            return mat_data
        except Exception as e:
            print(f"Error reading MATLAB file: {str(e)}")
            return None
    
    def setLabel(self, xlabel, ylabel, graph_name):
        self.x_label = xlabel
        self.y_label = ylabel
        self.graph = graph_name


    def resetLabels(self):
        self.filename = ""
        self.source = ""
        self.fignbr=""
        self.x_label= "X_Axis"
        self.y_label= "Y_Axis"
        self.graph = [] 

    def read_all_matfiles(self):
        dataframes = []
        for file_name in os.listdir(self.directory_path):
            if file_name.endswith(".mat"):
                file_path = os.path.join(self.directory_path, file_name)
                df = self.read_matfile(file_path)


    def create_plotly_figure(self, mat_data):
        if self.filename.startswith("xSOC"):
            self.create_xSOC_figure(mat_data)
        elif self.filename.startswith("Load"):
            self.create_General_figure(mat_data)
        else:
            print("Not implemented")

    def extract_data_in_df(self, mat_data):
        LegendVec = mat_data.get("Legend_Vec", [])
        x_axis_data_mat = mat_data.get('X_Axis_Data_Mat', np.array([]))
        # Check if X_Axis_Data_Mat is not empty by checking its shape instead of its truth value
        num_columns = x_axis_data_mat.shape[1] if x_axis_data_mat.ndim > 1 else 0

        x_axis_data = [[] for _ in range(num_columns)]
        y_axis_data = [[] for _ in range(num_columns)]
        y_axis_data_max = [[] for _ in range(num_columns)]
        y_axis_data_min = [[] for _ in range(num_columns)]

        # Assuming the structure of mat_data is as expected
        for i in range(len(x_axis_data_mat)):
            for j in range(num_columns):
                x_axis_data[j].append(x_axis_data_mat[i][j])
                y_axis_data[j].append(mat_data.get('Y_Axis_Data_Mat', np.array([]))[i][j])
                y_axis_data_max[j].append(mat_data.get('Y_Axis_Data_Max_Mat', np.array([]))[i][j])
                y_axis_data_min[j].append(mat_data.get('Y_Axis_Data_Min_Mat', np.array([]))[i][j])

        # Dynamically creating column names and data for the dataframe
        df_data = {}
        for j in range(num_columns):
            df_data[f'x_{LegendVec[0][j][0]}'] = x_axis_data[j]
            df_data[f'y_{LegendVec[0][j][0]}'] = y_axis_data[j]
            df_data[f'y_max_{LegendVec[0][j][0]}'] = y_axis_data_max[j]
            df_data[f'y_min_{LegendVec[0][j][0]}'] = y_axis_data_min[j]

        df = pd.DataFrame(df_data)

        return df

    def create_xSOC_figure(self, mat_data):
        
        x_axis_data_0 = []
        x_axis_data_1 = []
        x_axis_data_2 = []

        y_axis_data_0 = []
        y_axis_data_1 = []
        y_axis_data_2 = []

        y_axis_data_max_0 = []
        y_axis_data_max_1 = []
        y_axis_data_max_2 = []

        y_axis_data_min_0 = []
        y_axis_data_min_1 = []
        y_axis_data_min_2 = []

        LegendVec = mat_data.get("Legend_Vec", [])
        

        for i in range(0, len(mat_data.get('X_Axis_Data_Mat', []))-1):

            x_axis_data_0.append(mat_data.get('X_Axis_Data_Mat',[])[i][0])
            x_axis_data_1.append(mat_data.get('X_Axis_Data_Mat',[])[i][1])
            x_axis_data_2.append(mat_data.get('X_Axis_Data_Mat',[])[i][2])

            y_axis_data_0.append(mat_data.get('Y_Axis_Data_Mat',[])[i][0])
            y_axis_data_1.append(mat_data.get('Y_Axis_Data_Mat',[])[i][1])
            y_axis_data_2.append(mat_data.get('Y_Axis_Data_Mat',[])[i][2])

            y_axis_data_max_0.append(mat_data.get('Y_Axis_Data_Max_Mat',[])[i][0])
            y_axis_data_max_1.append(mat_data.get('Y_Axis_Data_Max_Mat',[])[i][1])
            y_axis_data_max_2.append(mat_data.get('Y_Axis_Data_Max_Mat',[])[i][2])

            y_axis_data_min_0.append(mat_data.get('Y_Axis_Data_Min_Mat',[])[i][0])
            y_axis_data_min_1.append(mat_data.get('Y_Axis_Data_Min_Mat',[])[i][1])
            y_axis_data_min_2.append(mat_data.get('Y_Axis_Data_Min_Mat',[])[i][2])

        ### FIGURE 1 ###
        trace1 = go.Scatter(x=x_axis_data_0, y=y_axis_data_0, mode='lines', name=self.graph)
        trace2 = go.Scatter(x=x_axis_data_0, y=y_axis_data_max_0, mode='lines', name=self.graph)
        trace3 = go.Scatter(x=x_axis_data_0, y=y_axis_data_min_0, mode='lines', name=self.graph)

        data = [trace1, trace2, trace3]

        layout = go.Layout(
            title=str(LegendVec[0][0]),
            xaxis=dict(title=self.x_label),
            yaxis=dict(title=self.y_label)
        )
        
        fig1 = go.Figure(data=data, layout=layout)
     

        ### FIGURE 2 ###
        trace1 = go.Scatter(x=x_axis_data_1, y=y_axis_data_1, mode='lines', name='Y_Axis_Data')
        trace2 = go.Scatter(x=x_axis_data_1, y=y_axis_data_max_1, mode='lines', name='Y_Axis_Data_Max')
        trace3 = go.Scatter(x=x_axis_data_1, y=y_axis_data_min_1, mode='lines', name='Y_Axis_Data_Min')

        data = [trace1, trace2, trace3]

        layout = go.Layout(
            title=str(LegendVec[0][1]),
            xaxis=dict(title=self.x_label),
            yaxis=dict(title='Y-Axis Label')
        )
        
        fig2 = go.Figure(data=data, layout=layout)
        
        ### FIGURE 3 ###
        trace1 = go.Scatter(x=x_axis_data_2, y=y_axis_data_2, mode='lines', name='Y_Axis_Data')
        trace2 = go.Scatter(x=x_axis_data_2, y=y_axis_data_max_2, mode='lines', name='Y_Axis_Data_Max')
        trace3 = go.Scatter(x=x_axis_data_2, y=y_axis_data_min_2, mode='lines', name='Y_Axis_Data_Min')

        data = [trace1, trace2, trace3]

        layout = go.Layout(
            title=str(LegendVec[0][2]),
            xaxis=dict(title='X-Axis Label'),
            yaxis=dict(title='Y-Axis Label')
        )
        
        fig3 = go.Figure(data=data, layout=layout)

        fig1.show()
        fig2.show()
        fig3.show()

    def create_LoadCollective_figure(self, mat_data):
        x_axis_data_0 = []
        x_axis_data_1 = []

        y_axis_data_0 = []
        y_axis_data_1 = []

        y_axis_data_max_0 = []
        y_axis_data_max_1 = []

        y_axis_data_min_0 = []
        y_axis_data_min_1 = []

        LegendVec = mat_data.get("Legend_Vec", [])
        

        for i in range(0, len(mat_data.get('X_Axis_Data_Mat', []))-1):

            x_axis_data_0.append(mat_data.get('X_Axis_Data_Mat',[])[i][0])
            x_axis_data_1.append(mat_data.get('X_Axis_Data_Mat',[])[i][1])

            y_axis_data_0.append(mat_data.get('Y_Axis_Data_Mat',[])[i][0])
            y_axis_data_1.append(mat_data.get('Y_Axis_Data_Mat',[])[i][1])

            y_axis_data_max_0.append(mat_data.get('Y_Axis_Data_Max_Mat',[])[i][0])
            y_axis_data_max_1.append(mat_data.get('Y_Axis_Data_Max_Mat',[])[i][1])

            y_axis_data_min_0.append(mat_data.get('Y_Axis_Data_Min_Mat',[])[i][0])
            y_axis_data_min_1.append(mat_data.get('Y_Axis_Data_Min_Mat',[])[i][1])

        ### FIGURE 1 ###
        trace1 = go.Scatter(x=x_axis_data_0, y=y_axis_data_0, mode='lines', name='Y_Axis_Data')
        trace2 = go.Scatter(x=x_axis_data_0, y=y_axis_data_max_0, mode='lines', name='Y_Axis_Data_Max')
        trace3 = go.Scatter(x=x_axis_data_0, y=y_axis_data_min_0, mode='lines', name='Y_Axis_Data_Min')

        data = [trace1, trace2, trace3]

        layout = go.Layout(
            title=str(LegendVec[0][0]),
            xaxis=dict(title='X-Axis Label'),
            yaxis=dict(title='Y-Axis Label')
        )
        
        fig1 = go.Figure(data=data, layout=layout)
     

        ### FIGURE 2 ###
        trace1 = go.Scatter(x=x_axis_data_1, y=y_axis_data_1, mode='lines', name='Y_Axis_Data')
        trace2 = go.Scatter(x=x_axis_data_1, y=y_axis_data_max_1, mode='lines', name='Y_Axis_Data_Max')
        trace3 = go.Scatter(x=x_axis_data_1, y=y_axis_data_min_1, mode='lines', name='Y_Axis_Data_Min')

        data = [trace1, trace2, trace3]

        layout = go.Layout(
            title=str(LegendVec[0][1]),
            xaxis=dict(title='X-Axis Label'),
            yaxis=dict(title='Y-Axis Label')
        )
        
        fig2 = go.Figure(data=data, layout=layout)
        
        fig1.show()
        fig2.show()

    def create_General_figure(self, mat_data):

        LegendVec = mat_data.get("Legend_Vec", [])
        #print(len(LegendVec[0]))
        x_axis_data = [[] for i in range(len(LegendVec[0])+1)] 

        y_axis_data = [[] for i in range(len(LegendVec[0])+1)] 

        y_axis_data_max = [[] for i in range(len(LegendVec[0])+1)] 

        y_axis_data_min = [[] for i in range(len(LegendVec[0])+1)] 
        y_axis_data_mean = [[] for i in range(len(LegendVec[0])+1)] 
        
        alltogether_data = []
        for j in range(0, len(LegendVec[0])):
            graph_name = ""
            print(len(self.graph))
            print(len(LegendVec[0]))
            if len(self.graph) == len(LegendVec[0]):
                print("yes")
                graph_name = self.graph[j]
                print(graph_name)
            
            x_axis_data[j] = []
            y_axis_data[j] = []
            y_axis_data_max[j] = []
            y_axis_data_min[j] = []
            y_axis_data_mean[j] = []

            for i in range(0, len(mat_data.get('X_Axis_Data_Mat', []))):
                
                x_axis_data[j].append(mat_data.get('X_Axis_Data_Mat',[])[i][j])
                y_axis_data[j].append(mat_data.get('Y_Axis_Data_Mat',[])[i][j])
                y_axis_data_max[j].append(mat_data.get('Y_Axis_Data_Max_Mat',[])[i][j])
                y_axis_data_min[j].append(mat_data.get('Y_Axis_Data_Min_Mat',[])[i][j])
                y_axis_data_mean[j].append(statistics.mean([mat_data.get('Y_Axis_Data_Mat',[])[i][j], mat_data.get('Y_Axis_Data_Max_Mat',[])[i][j], mat_data.get('Y_Axis_Data_Min_Mat',[])[i][j]]) )


            


            ### FIGURE ###
            trace1 = go.Scatter(x=x_axis_data[j], y=y_axis_data[j], mode='lines', name=graph_name )
            trace2 = go.Scatter(x=x_axis_data[j], y=y_axis_data_max[j], mode='markers', name=str(graph_name) + "_MAX")
            trace3 = go.Scatter(x=x_axis_data[j], y=y_axis_data_min[j], mode='markers', name=str(graph_name) + "_MIN")
            trace4 = go.Scatter(x=x_axis_data[j], y=y_axis_data_mean[j], mode='lines', name=str(graph_name) )

            data = [trace1, trace2, trace3]

            layout = go.Layout(
                title=str(str(LegendVec[0][j]) + " , " + str(self.source) + str(": Fig. ") + str(self.fignbr)),
                xaxis=dict(title=self.x_label),
                yaxis=dict(title=self.y_label)
            )
            alltogether_data.append(trace1)
            fig = go.Figure(data=data, layout=layout)
            fig.show()
        layout = go.Layout(
                title=str(str("Full Plot") + " , " + str(self.source) + str(": Fig. ") + str(self.fignbr)),
                xaxis=dict(title=self.x_label),
                yaxis=dict(title=self.y_label)
            )
        alltogether_fig = go.Figure(data=alltogether_data, layout=layout)
        alltogether_fig.show()
     
    def create_EIS_figure(self, mat_data):
        print(type(mat_data))
        print(mat_data.keys())
        print(mat_data['EIS_Parameter'])
        print('')
        print(mat_data['__header__'])
        print('')
        print(mat_data['__globals__'])
        print('')
        print(mat_data['EIS_Parameter'][0][0][0][0][0])
        print(mat_data['EIS_Parameter'][0][1][0][0][0])
        print(mat_data['EIS_Parameter'][0][2][0][0][0])
        print(mat_data['EIS_Parameter'][0][3][0][0][0])
        print(type(mat_data['EIS_Parameter'].dtype))


                
# if __name__ == '__main__':

#     # Specify the path to your MATLAB file
#     cwd = os.getcwd()
#     subdirectory = "mat_data"

#     directory_path = os.path.join(cwd, subdirectory)

#     file_name = "EIS_40_C_50SOC_80DOD_1C_1C.mat"

#     # Create an instance of the MatlabFileReader class

#     reader = MatlabFileReader(directory_path)
#     #reader.read_all_matfiles()


#     data = reader.read_matfile(file_name)
#     reader.create_EIS_figure(data)
