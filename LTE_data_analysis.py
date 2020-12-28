from os import path
import time
import datetime
import math
from dateutil import parser
import pandas as pd
import numpy as np
import os


def LTE(Data1, TRAIN):
    print(Data1)
    print(TRAIN)
    # decleare functions
    def _make_data_frame(myfile, desired_line_number, parameter):
        df = pd.DataFrame()
        line = myfile[desired_line_number]
        df["Line_Num"] = [desired_line_number]
        i = 0
        while line.strip() != '' and i <= parameter:
            line = myfile[desired_line_number]
            start_index = line.find(" : ")
            end_index = start_index + len(" : ")
            frame = line[0:start_index]
            data = line[end_index:]
            df[frame] = [data]
            desired_line_number = desired_line_number + 1
            i = i + 1
        return df
        print("make")
    
    def _get_samples(desired_line_number, samples_num, start_line_number, num_parameter_in_sample, myfile):
        df = pd.DataFrame()
        line = myfile[start_line_number]
        end = start_line_number + (samples_num * num_parameter_in_sample)
        j = start_line_number
        df["Line_Num"] = [desired_line_number]
        while j <= end - 1:
            for i in range(0, samples_num, 1):
                for k in range(0, num_parameter_in_sample, 1):
                    samp = j
                    line = myfile[samp]
                    if myfile[j] == '\n' or j > end or myfile[j] == '\t\n':
                        break
                    else:
                        start_index = line.find(" : ")
                        end_index = start_index + len(" : ")
                        Frame = line[0:start_index] + "[" + str(i) + "]"
                        Data = line[end_index:]
                        df[Frame] = [Data]
                        j = j + 1
                if myfile[j] == '\n' or j > end or myfile[j] == '\t\n':
                    break
            if myfile[j] == '\n' or j > end or myfile[j] == '\t\n':
                break
        return df
        print("_samples")
    
    ## get data frame according to Message type
    def _find_msg(msgtype, combination):
        msg = combination
        msg['C'] = np.where(msgtype == msg['Message_Type'], 'True', 'False')
        msg = msg[msg['C'] == "True"]
        msg = msg.iloc[:, :-1].reset_index()
        return msg
        print("find_msg")
    
    def _active_extract(Start, end, msg):
        msg['C'] = np.where((Start <= msg["HW Timestamp"]) & (end >= msg["HW Timestamp"]), 'True', 'False')
        msg = msg[msg['C'] == "True"]
        msg = msg.iloc[:, :-1].reset_index()
        return msg
        print("active_extract")
    
    def _find_msg_total(msgtype1, msgtype2, msgtype3, msgtype4, combination):
        msg = combination
        msg['C'] = np.where((msgtype1 == msg['Message_Type']) | (msgtype2 == msg['Message_Type']) \
                            | (msgtype3 == msg['Message_Type']) | (msgtype4 == msg['Message_Type']), 'True', 'False')
        msg = msg[msg['C'] == "True"]
        msg = msg.iloc[:, :-1]
        return msg
        print("find_msg_total")
    # print("find_msg_total")
    
    def _time_extract(active_time):
        time = active_time.filter(regex="HW Timestamp")
        for i in range(0, time.shape[0], 1):
            data = time["HW Timestamp"][i]
            start_index = data.find("(") + 1
            end_index = data.find("m") - 1
            data = float(data[start_index:end_index])
            time.at[i, 'HW Timestamp'] = data
        active_time.drop("HW Timestamp", 1, inplace=True)
        active_time = time.join(active_time)
        return active_time
        print("time_extract")
    
    ## search func
    ## i/p: txt file & the string i want to search for
    ## o/p: the matched string and its line_number
    def _search_multiple_strings_in_file(file_name, list_of_strings):
        """Get line from the file along with line numbers, which contains any string from the list"""
        line_number = 0
        list_of_results = []
        # Open the file in read only mode
        with open(file_name, 'r') as read_obj:
            # Read all lines in the file one by one
            for line in read_obj:
                line_number += 1
                # For each line, check if line contains any string from the list of strings
                for string_to_search in list_of_strings:
                    if string_to_search in line:
                        # If any string is found in line, then append that line along with line number in list
                        list_of_results.append((string_to_search, line_number, line.rstrip()))
        # Return list of tuples containing matched string, line numbers and lines where string is found
        return list_of_results
        print("search_multiple_strings_in_file")
    
    def _get_ack(start_line_number, myfile):
        df = pd.DataFrame()
        line_number = start_line_number
        i = 0
        k = 0
        while myfile[line_number] != '\t\n':
            line = myfile[line_number]
            if line == "    PhichInfo : \n":
                i = i + 1
                L1 = myfile[line_number - 2]
                start_index = L1.find(" : ")
                end_index = start_index + len(" : ")
                frame = L1[0:start_index] + "[" + str(k) + "]"
                data = L1[end_index:]
                df[frame] = [data]
                L2 = myfile[line_number - 1]
                start_index = L2.find(" : ")
                end_index = start_index + len(" : ")
                frame = L2[0:start_index] + "[" + str(k) + "]"
                data = L2[end_index:]
                df[frame] = [data]
                L3 = myfile[line_number + 6]
                start_index = L3.find(" : ")
                end_index = start_index + len(" : ")
                frame = L3[0:start_index - 4] + "[" + str(k) + "]"
                data = L3[end_index:]
                df[frame] = [data]
                k = k + 1
                df["Line_Num"] = start_line_number
            line_number = line_number + 1
        return df
        print("get_ack")
    
    """---------------Calculations-------------------------"""
    
    def _calc(file_naming, value_naming):
        file_naming = file_naming.replace(regex=["dBm"], value='')
        file_naming = file_naming.replace(regex=["dB"], value='')
        file_naming = file_naming.replace(regex=["Rank"], value='')
        final = pd.DataFrame(columns=['mean', 'var'])
        j = 0
        num = 0
        sam_first = 0
        sam_second = 0
        size = file_naming.shape[0]
        for i in range(0, size - 1, 1):
            T1 = file_naming['Time'].iat[j]
            T2 = file_naming['Time'].iat[i + 1]
            obj1 = parser.parse(T1)
            obj2 = parser.parse(T2)
            diff = obj2 - obj1
            t1 = parser.parse("00:00:00.0")
            t2 = parser.parse("00:00:01.0")
            threshold = t2 - t1
            if diff > threshold:
                new = file_naming.iloc[j: i + 1]
                if (value_naming == "Throughput" or value_naming == "Total_Block_Size"):
                    time1 = new["Time"].iat[0]
                    time2 = new["Time"].iat[-1]
                    t11 = parser.parse(time1)
                    t22 = parser.parse(time2)
                    file_duration = datetime.timedelta.total_seconds(t22 - t11)
                    if new.shape[0] == 1:
                        sam_first = (new[value_naming][j] * 1e-3) / (new["HW Timestamp"].iat[0] * 1e-3)  # Throughput
                        sam_second = new[value_naming][j]
                        sam_second = sam_second.mean()
                    else:
                        sam_first = (new[value_naming].sum() * 1e-3) / (file_duration)  #
                        sam_second = new[value_naming].sum()
                        sam_second = sam_second.mean()
                    final = final.append({'mean': sam_first, 'var': sam_second}, ignore_index=True)  # Throughput
                elif value_naming == "Number of Samples":
                    sam_first = new[value_naming].sum() / 10
                    final = final.append({'mean': sam_first}, ignore_index=True)  # scheduling
                else:
                    sam_first = new[value_naming].astype(float).mean()  # scheduling
                    sam_second = new[value_naming].astype(float).var()  # scheduling
                    final = final.append({'mean': sam_first, 'var': sam_second}, ignore_index=True)
                num = num + 1
                i = i + 1
                j = i
                sam_first = 0
                sam_second = 0
        """
        s = file_naming.iloc[j: file_naming.shape[0]]
        if (value_naming == "Throughput" or value_naming == "Total_Block_Size"):
            time11 = s["Time"].iat[0]
            time22 = s["Time"].iat[-1]
            t111 = parser.parse(time11)
            t222 = parser.parse(time22)
            file_duration2 = datetime.timedelta.total_seconds(t222-t111)
            if s.shape[0] == 1:
                var_first = (s[value_naming][j]* 1e-3) / (s["HW Timestamp"].iat[0]*1e-3)#Throughput
                var_second = s[value_naming][j]
                var_second = var_second.mean()
            else:
                var_first = (s[value_naming].sum()* 1e-3) / (file_duration2)#
                var_second = s[value_naming].sum()
                var_second = var_second.mean()
            final = final.append({'mean': var_first, 'var': var_second}, ignore_index=True)#Thrsoughput
    
        elif value_naming == "Number of Samples":
            var_first = s[value_naming].sum()/10
            final = final.append({'mean': var_first}, ignore_index=True)#scheduling
    
        else:
            m1 = s[value_naming].astype(float).mean()
            m2 = s[value_naming].astype(float).var()
            #    num1=s.shape[0]
            final = final.append({'mean': m1, 'var': m2}, ignore_index=True)
            """
        return final
        print("calc")
    
    def _files(file_number):
        file_num = file_number
        active_data = pd.read_csv(os.path.dirname(file_num)+'PDSCH_Demapper_Active_Time' + os.path.basename(file_num) + '.csv')
        active_intra = pd.read_csv(os.path.dirname(file_num)+'Intra_Frequency_Measurement_Active_Time' + os.path.basename(file_num) + '.csv')
        active_serving = pd.read_csv(os.path.dirname(file_num)+'Serving_Cell_Measurement_Active_Time' + os.path.basename(file_num)+ '.csv')
        active_log = pd.read_csv(os.path.dirname(file_num)+'LL1_PUSCH_CSF_Log_Active_Time' + os.path.basename(file_num)+ '.csv')
        active_phich = pd.read_csv(os.path.dirname(file_num)+'PDCCH_PHICH_Indication_Active_Time' + os.path.basename(file_num)+ '.csv')
        active_ll1 = pd.read_csv(os.path.dirname(file_num)+'LL1_Serving_Cell_Measurement' + os.path.basename(file_num)+ '.csv')
    
        return active_data, active_intra, active_serving, active_log, active_phich, active_ll1
        print("files")
    
    def _summary(active_data, active_intra, active_serving, active_log, active_phich, active_ll1, file_number):
    
        active_LL1 = active_ll1.replace(regex=["MHz"], value='')
        demapper1_mod = active_data.filter(regex='Modulation for Stream ')
        demapper1_rank = active_data.filter(regex='Spatial Rank')
        demapper1_mod = demapper1_mod.transpose()
        demapper1_rank = demapper1_rank.transpose()
        demapper1_mod_type = pd.DataFrame(index=range(0, len(demapper1_mod.columns)))
        demapper1_rank_type = pd.DataFrame(index=range(0, len(demapper1_rank.columns)))
    
        i = -1
        for col1 in demapper1_mod.columns:
            i = i + 1
            count1 = demapper1_mod[col1].value_counts(ascending=True)
            for name1 in count1.index:
                demapper1_mod_type.at[i, name1] = count1[name1]
    
        j = -1
        for col2 in demapper1_rank.columns:
            j = j + 1
            count2 = demapper1_rank[col2].value_counts(ascending=True)
            for name2 in count2.index:
                demapper1_rank_type.at[j, name2] = count2[name2]
    
        list_heads_m = list(demapper1_mod_type.columns)
        list_heads_r = list(demapper1_rank_type.columns)
        demapper1_mod_type.columns = demapper1_mod_type.columns.str.strip()
        demapper1_rank_type.columns = demapper1_rank_type.columns.str.strip()
        list_heads_m = list(demapper1_mod_type.columns)
        list_heads_r = list(demapper1_rank_type.columns)
    
        for col11 in list_heads_m:
            if col11 == '16QAM':
                demapper1_mod_type['16QAM'] = demapper1_mod_type['16QAM'] * 4.0
            elif col11 == '64QAM':
                demapper1_mod_type['64QAM'] = demapper1_mod_type['64QAM'] * 6
            elif col11 == '256QAM':
                demapper1_mod_type['256QAM'] = demapper1_mod_type['256QAM'] * 8
            elif col11 == 'QPSK':
                demapper1_mod_type['QPSK'] = demapper1_mod_type['QPSK'] * 2
    
        for col22 in list_heads_r:
            if col22 == 'Rank 2':
                demapper1_rank_type['Rank 2'] = demapper1_rank_type['Rank 2'] * 2
    
        demapper1_mod_type = demapper1_mod_type.fillna(0)
        demapper1_rank_type = demapper1_rank_type.fillna(0)
    
        demapper1_mod_type['sum'] = demapper1_mod_type.sum(axis=1)
        demapper1_rank_type['sum'] = demapper1_rank_type.sum(axis=1)
        demapper1_mod_type['Avg.No Of bits'] = (demapper1_mod_type['sum'] / active_data['Number of Samples']) / 2
        demapper1_rank_type['Rank'] = (demapper1_rank_type['sum'] / active_data['Number of Samples'])
    
        active_data['Avg.No Of bits'] = demapper1_mod_type['Avg.No Of bits']
        active_data['Rank'] = demapper1_rank_type['Rank']
        ###############################################################################
        demapper1_rb = active_data.filter(regex='RB Allocation Slot 0 ')
        demapper1_rb_no_ones = demapper1_rb.fillna('0x0')
        demapper1_rb_no_ones = demapper1_rb_no_ones.transpose()
        Demapper1_length = len(demapper1_rb.columns)
        no_of_ones_list = []
        no_of_ones = 0
        for col in demapper1_rb_no_ones.columns:
            no_of_ones_list.append(no_of_ones)
            no_of_ones = 0
            for i in range(0, Demapper1_length, 1):
                demapper1_rb_no_ones[col][i] = bin(int(demapper1_rb_no_ones[col][i], 16)).count('1')
                no_of_ones = no_of_ones + demapper1_rb_no_ones[col][i]
    
        no_of_ones_list.append(no_of_ones)
        no_of_ones_list.pop(0)
        no_of_ones_dataframe = pd.DataFrame(no_of_ones_list, columns=['ones'])
        no_of_ones_dataframe['ones'] = no_of_ones_dataframe['ones'] / active_data['Number of Samples']
        active_data['no of RBs'] = no_of_ones_dataframe['ones']
        ###############################################################################
        demapper_sub = active_data.filter(regex='    Subframe Number')
        demapper_frame = active_data.filter(regex='    System Frame Number')
        data_stream1 = active_data.filter(regex='    Transport Block Size for Stream 0')
        data_stream2 = active_data.filter(regex='    Transport Block Size for Stream 1')
        index = active_data.filter(regex='index')
        time = active_data.filter(regex='Time')
        ack_frame = active_phich.filter(regex='    PDCCH Timing System Frame Number')
        ack_sub = active_phich.filter(regex='    PDCCH Timing Subframe Number')
        ack_value = active_phich.filter(regex='          PHICH Value')
    
        size1 = active_data.shape[0]
        size2 = data_stream1.shape[1]
    
        new1 = pd.DataFrame()
        new2 = pd.DataFrame()
        data_stream_1 = pd.DataFrame()
        data_stream_2 = pd.DataFrame()
        for i in range(0, size1, 1):
            col_frame = demapper_frame.iloc[i]
            col_sub = demapper_sub.iloc[i]
            data1 = data_stream1.iloc[i]
            data2 = data_stream2.iloc[i]
            new1 = pd.concat([new1, col_frame], sort=False)
            new2 = pd.concat([new2, col_sub], sort=False)
            data_stream_1 = pd.concat([data_stream_1, data1], sort=False)
            data_stream_2 = pd.concat([data_stream_2, data2], sort=False)
        new1 = new1.reset_index()
        new2 = new2.reset_index()
        data_stream_1 = data_stream_1.reset_index()
        data_stream_2 = data_stream_2.reset_index()
        Index = pd.DataFrame(np.repeat(index.values, size2, axis=0))
        time = pd.DataFrame(np.repeat(time.values, size2, axis=0))
    
        new1.rename(columns={list(new1)[1]: 'Frame'}, inplace=True)
        new2.rename(columns={list(new2)[1]: 'Sub_Frame'}, inplace=True)
        data_stream_1.rename(columns={list(data_stream_1)[1]: 'stream[0]'}, inplace=True)
        data_stream_2.rename(columns={list(data_stream_2)[1]: 'stream[1]'}, inplace=True)
        Index.rename(columns={list(Index)[0]: 'index_Num'}, inplace=True)
        time.rename(columns={list(time)[1]: 'Time'}, inplace=True)
    
        horizontal_stack = pd.concat([time['Time'], Index['index_Num'], new1['Frame'],
                                      new2['Sub_Frame'], data_stream_1['stream[0]'], data_stream_2['stream[1]']],
                                     axis=1)
        final_stack = horizontal_stack.dropna()
        # final_stack['SFN'] = final_stack['Frame'].map(str) + '_' + final_stack['Sub_Frame'].map(str)
        final_stack['SFN'] = final_stack['Frame'].map(str).str.cat(final_stack['Sub_Frame'].map(str), sep="_")
        #####################################################################
        size11 = active_phich.shape[0]
        frame_ack = pd.DataFrame()
        ack_subframe = pd.DataFrame()
        value = pd.DataFrame()
    
        for j in range(0, size11, 1):
            col_ack_frame = ack_frame.iloc[j]
            col_ack_sub = ack_sub.iloc[j]
            col_ack_value = ack_value.iloc[j]
    
            frame_ack = pd.concat([frame_ack, col_ack_frame], sort=False)
            ack_subframe = pd.concat([ack_subframe, col_ack_sub], sort=False)
            value = pd.concat([value, col_ack_value], sort=False)
    
        frame_ack = frame_ack.reset_index()
        ack_subframe = ack_subframe.reset_index()
        value = value.reset_index()
        frame_ack.rename(columns={list(frame_ack)[1]: 'Frame'}, inplace=True)
        ack_subframe.rename(columns={list(ack_subframe)[1]: 'Sub_Frame'}, inplace=True)
        value.rename(columns={list(value)[1]: 'PHICH_value'}, inplace=True)
        horizontal_ack_stack = pd.concat([frame_ack['Frame'], ack_subframe['Sub_Frame'], value['PHICH_value']], axis=1)
        final_ack_stack = horizontal_ack_stack.dropna()
        # final_ack_stack['SFN'] = final_ack_stack['Frame'].map(str) + '_' + final_ack_stack['Sub_Frame'].map(str)
        final_ack_stack['SFN'] = final_ack_stack['Frame'].map(str).str.cat(final_ack_stack['Sub_Frame'].map(str),
                                                                           sep="_")
        time = pd.concat([time['Time'], Index['index_Num']], axis=1)
        ####################################################################
        nnnn = pd.DataFrame()
        ff0 = pd.DataFrame()
        new___frames = pd.DataFrame()
        new_frames = pd.merge(final_stack, final_ack_stack, how="inner", on="SFN")
        ack_final = new_frames.dropna()
        acked = ack_final[ack_final.PHICH_value != 'PHICH ACK']
        ff0 = acked.groupby('index_Num')['stream[0]'].sum().reset_index()
        ff1 = acked.groupby('index_Num')['stream[1]'].sum().reset_index()
        ff0['ACK_Throughput'] = ff0['stream[0]'] + ff1['stream[1]']
        final_ack = pd.merge(time, ff0, how="inner", on="index_Num")
        final_ack = final_ack.drop_duplicates().reset_index()
        final_ack = final_ack.drop(final_ack.filter(regex="stream").columns, axis=1)
        final_ack = final_ack.drop(final_ack.filter(regex="index").columns, axis=1)
        new___frames = pd.merge(ff0, active_data, how="outer", left_on="index_Num", right_on="index")
        nnnn = new___frames
        nnnn = nnnn.fillna(0)
        active_data['Throughput'] = nnnn['ACK_Throughput']
        ###############################################################################
        serving_filtered_rsrp = _calc(active_intra, "Serving Filtered RSRP (dBm)")
        inst_measured_rsrq = _calc(active_serving, "          Inst RSRQ")
        inst_measured_sinr_rx0 = _calc(active_serving, "          SINR Rx[0]")
        inst_measured_sinr_rx1 = _calc(active_serving, "          SINR Rx[1]")
        wideband_cqi_cw0 = _calc(active_log, "  WideBand CQI CW0")
        #    WideBand_CQI_CW1 = _calc(active_log, "  WideBand CQI CW1")
        B_W = _calc(active_LL1, "Measurement Bandwidth")
        mobile_rank = _calc(active_log, "  Rank Index")
        no_of_rbs = _calc(active_data, "no of RBs")
        total_throughput = _calc(active_data, "Total_Block_Size")
        ack_throughput = _calc(active_data, "Throughput")
        scheduling = _calc(active_data, "Number of Samples")
        network_rank = _calc(active_data, "Rank")
        modulation = _calc(active_data, "Avg.No Of bits")
        ###############################################################################
        sch = np.array(scheduling["mean"].values.tolist())
        scheduling["mean"] = np.where(sch > 100.0, 100, sch).tolist()
        ###############################################################################
        for i in range(0, no_of_rbs.shape[0], 1):
            no_of_rbs['mean'][i] = math.ceil(no_of_rbs['mean'][i])
        ###############################################################################
        # ensure that all files have the same length
        intra_size = serving_filtered_rsrp.shape[0]
        log_size = wideband_cqi_cw0.shape[0]
        serving_size = inst_measured_rsrq.shape[0]
        data_size = total_throughput.shape[0]
        bw_size = B_W.shape[0]
        if data_size != serving_size or data_size != intra_size or data_size != log_size or data_size != bw_size:
            max_value = max(intra_size, serving_size, log_size, data_size, bw_size)
            for i in range(max_value - intra_size):
                serving_filtered_rsrp = serving_filtered_rsrp.append(
                    {"mean": serving_filtered_rsrp["mean"].iat[-1], "var": serving_filtered_rsrp["var"].iat[-1]},
                    ignore_index=True, sort=False)
            for j in range(max_value - serving_size):
                inst_measured_rsrq = inst_measured_rsrq.append({"mean": inst_measured_rsrq["mean"].iat[-1],
                                                                "var": inst_measured_rsrq["var"].iat[-1]},
                                                               ignore_index=True, sort=False)
                inst_measured_sinr_rx0 = inst_measured_sinr_rx0.append({"mean": inst_measured_sinr_rx0["mean"].iat[-1],
                                                                        "var": inst_measured_sinr_rx0["var"].iat[-1]},
                                                                       ignore_index=True, sort=False)
                inst_measured_sinr_rx1 = inst_measured_sinr_rx1.append({"mean": inst_measured_sinr_rx1["mean"].iat[-1],
                                                                        "var": inst_measured_sinr_rx1["var"].iat[-1]},
                                                                       ignore_index=True, sort=False)
            for k in range(max_value - log_size):
                wideband_cqi_cw0 = wideband_cqi_cw0.append({"mean": wideband_cqi_cw0["mean"].iat[-1],
                                                            "var": wideband_cqi_cw0["var"].iat[-1]}, ignore_index=True,
                                                           sort=False)
                #            WideBand_CQI_CW1 = WideBand_CQI_CW1.append({"mean":WideBand_CQI_CW1["mean"].iat[-1],
                #                                                        "var":WideBand_CQI_CW1["var"].iat[-1]}, ignore_index=True, sort=False)
                mobile_rank = mobile_rank.append({"mean": mobile_rank["mean"].iat[-1],
                                                  "var": mobile_rank["var"].iat[-1]}, ignore_index=True, sort=False)
            for l in range(max_value - bw_size):
                B_W = B_W.append({"mean": B_W["mean"].iat[-1]}, ignore_index=True, sort=False)
    
        ###############################################################################
        T1 = active_data['Time'].iat[0]
        T2 = active_data['Time'].iat[-1]
        obj1 = parser.parse(T1)
        obj2 = parser.parse(T2)
        #        obj1=X["Time"].iat[j]
        #        obj2=X["Time"].iat[i+1]
        file_duration = datetime.timedelta.total_seconds(obj2 - obj1)
    
        summary_report = pd.DataFrame()
        for i in range(0, no_of_rbs.shape[0], 1):
            summary_report = summary_report.append({"File": thefilepath,
                                                    "File duration[sec]": file_duration,
                                                    "BW": B_W["mean"][i],
                                                    "Serving Filtered RSRP mean[dBm]": serving_filtered_rsrp["mean"][i],
                                                    "Inst RSRQ mean [dB]": inst_measured_rsrq["mean"][i],
                                                    "Inst SINR RX0 mean [dB]": inst_measured_sinr_rx0["mean"][i],
                                                    "Inst SINR_RX1 mean [dB]": inst_measured_sinr_rx1["mean"][i],
                                                    "WideBand CQI_CW0 mean": wideband_cqi_cw0["mean"][i],
                                                    # "WideBand CQI_CW1 mean":WideBand_CQI_CW1["mean"][i],
                                                    "Network rank mean": network_rank["mean"][i],
                                                    "Mobile rank mean": mobile_rank["mean"][i],
                                                    "Modulation bits mean": modulation["mean"][i],
                                                    "No of RBs [bits]": no_of_rbs["mean"][i],
                                                    "Scheduling_percentage[%]": str(scheduling["mean"][i]) + ' %',
                                                    "Total Transport block size mean": total_throughput["var"][i],
                                                    "Acked Transport block size mean": ack_throughput["var"][i],
                                                    "Total Throughput[kbit/s]": total_throughput["mean"][i],
                                                    "Acked Throughput[kbit/s]": ack_throughput["mean"][i]},
                                                   ignore_index=True, sort=False)
        return summary_report
        print("summary")
    
    ###############################################################################
    ###############################################################################
    # main part
    def _Analysis_(thefilepath):
        print(thefilepath)
        thefilepath = thefilepath
        final1 = pd.DataFrame(columns=['Line_Num', 'Message_Type'])
        final2 = pd.DataFrame(columns=['Line_Num', 'Time'])
        matched_lines1 = _search_multiple_strings_in_file(thefilepath, ['Mode Report Message type'])
        matched_lines2 = _search_multiple_strings_in_file(thefilepath, ["Time : "])
    
        matched_lines2 = [i for i in matched_lines2 if i[2] >= "Time : "]
    
        ## list all message types
        for elem1 in matched_lines1:
            final1 = final1.append({'Message_Type': elem1[2], 'Line_Num': elem1[1]}, ignore_index=True)
        final1 = final1.reset_index()
    
        ## list all time stampes
        for elem2 in matched_lines2:
            final2 = final2.append({'Time': elem2[2], 'Line_Num': elem2[1] - 1}, ignore_index=True)
        final2 = final2.reset_index()
        combination = final1.replace(regex=['Mode Report Message type: '], value='')
        combination = combination.replace(regex=["\*"], value='')
        combination = combination.dropna()
        # combination.to_csv('combination1.csv', index=True)
    
        ######################################################
        ## get unique messages
        #    unique_message = list(combination.Message_Type.unique())
        #    unique_message_num = combination.Message_Type.nunique()
        ######################
        myfile = []  # Declare an empty list
        with open(thefilepath, 'rt') as my_file:  # Open lorem.txt for reading text data.
            for myline in my_file:  # For each line, stored as myline,
                myfile.append(myline)  # add its contents to mylines.
    
        # TotalMsg = FindMsgTotal(" LL1 PDSCH Demapper Configuration"," LL1 PUSCH CSF Log"," ML1 Serving Cell Measurement Result"," ML1 Connected Mode LTE intra-Frequency Measurement Results",combination)
        # TotalMsg.to_csv('TotalMsg'+file_number+'.csv', index=True)
        pdsch_demapper = _find_msg(" LL1 PDSCH Demapper Configuration", combination)
        pusch_csf_log = _find_msg(" LL1 PUSCH CSF Log", combination)
        serving_cell_measurement = _find_msg(" ML1 Serving Cell Measurement Result", combination)
        intra_frequency_measurement = _find_msg(" ML1 Connected Mode LTE Intra-Frequency Measurement Results",
                                                combination)
        pdcch_phich_indication = _find_msg(" ML1 PDCCH-PHICH Indication Report", combination)
        LL1_Serving_Cell_Measurement = _find_msg(" LL1 Serving Cell Measurement Results", combination)
        ll1_pusch_csf_log_df = pd.DataFrame()
        LL1_pusch_csf_log_size = pusch_csf_log.shape[0]
        for i in range(0, LL1_pusch_csf_log_size, 1):
            desired_line_number = int(pusch_csf_log["Line_Num"][i]) + 2
            df_log = _make_data_frame(myfile, desired_line_number, 49)
            ll1_pusch_csf_log_df = ll1_pusch_csf_log_df.append(df_log, ignore_index=True, sort=False)
        ll1_pusch_csf_log_df = _time_extract(ll1_pusch_csf_log_df)
        # ll1_pusch_csf_log_df.to_csv('ll1_pusch_csf_log_df'+file_number+'.csv', index=True)
        serving_cell_measurement_df = pd.DataFrame()
        serving_cell_measurement_size = serving_cell_measurement.shape[0]
        for i in range(0, serving_cell_measurement_size, 1):
            desired_line_number2 = int(serving_cell_measurement["Line_Num"][i]) + 2
            df_measure = _make_data_frame(myfile, desired_line_number2, 55)
            serving_cell_measurement_df = serving_cell_measurement_df.append(df_measure, ignore_index=True, sort=False)
        serving_cell_measurement_df = _time_extract(serving_cell_measurement_df)
        # serving_cell_measurement_df.to_csv('serving_cell_measurement_df'+file_number+'.csv', index=True)
    
        intra_frequency_measurement_df = pd.DataFrame()
        intra_frequency_measurement_size = intra_frequency_measurement.shape[0]
        for i in range(0, intra_frequency_measurement_size, 1):
            desired_line_number3 = int(intra_frequency_measurement["Line_Num"][i]) + 2
            df_intra = _make_data_frame(myfile, desired_line_number3, 16)
            intra_frequency_measurement_df = intra_frequency_measurement_df.append(df_intra, ignore_index=True,
                                                                                   sort=False)
    
        intra_frequency_measurement_samp_df = pd.DataFrame()
        intra_frequency_measurement_samp_size = intra_frequency_measurement.shape[0]
        for i in range(0, intra_frequency_measurement_samp_size, 1):
            start_line_number2 = int(intra_frequency_measurement_df["Line_Num"][i]) + 17
            samples_num2 = int(intra_frequency_measurement_df["Number Of Measured Neighbor Cells"][i])
            L = int(intra_frequency_measurement_df["Line_Num"][i])
            intra_samp = _get_samples(L, samples_num2, start_line_number2, 4, myfile)
            intra_frequency_measurement_samp_df = intra_frequency_measurement_samp_df.append(intra_samp,
                                                                                             ignore_index=True,
                                                                                             sort=False)
        intra_frequency_measurement_df = pd.merge(intra_frequency_measurement_df,
                                                  intra_frequency_measurement_samp_df, on="Line_Num", how="outer")
        intra_frequency_measurement_df["Detected Physical Cells"] = "\t\n"
    
        intra_frequency_measurement_second_samp_df = pd.DataFrame()
        #    intra_frequency_measurement_samp2_size = intra_frequency_measurement.shape[0]
        #    intra_samp22 = pd.DataFrame()
        for i in range(0, intra_frequency_measurement_size, 1):
            samples_num22 = int(intra_frequency_measurement_df["Number Of Detected Cells"][i])
            start_line_number22 = (int(intra_frequency_measurement_df["Line_Num"][i]) + 18) \
                                  + (int(intra_frequency_measurement_df["Number Of Measured Neighbor Cells"][i]) * 4)
            L = int(intra_frequency_measurement_df["Line_Num"][i])
            intra_samp2 = _get_samples(L, samples_num22, start_line_number22, 5, myfile)
            intra_frequency_measurement_second_samp_df = intra_frequency_measurement_second_samp_df.append(intra_samp2,
                                                                                                           ignore_index=True,
                                                                                                           sort=False)
        intra_frequency_measurement_df = pd.merge(intra_frequency_measurement_df,
                                                  intra_frequency_measurement_second_samp_df, on="Line_Num",
                                                  how="outer")
        intra_frequency_measurement_df = _time_extract(intra_frequency_measurement_df)
        # intra_frequency_measurement_df.to_csv('intra_frequency_measurement_df'+file_number+'.csv', index=True)
    
        pdsch_demapper_df = pd.DataFrame()
        pdsch_demapper_size = pdsch_demapper.shape[0]
        for i in range(0, pdsch_demapper_size, 1):
            desired_line_number4 = int(pdsch_demapper["Line_Num"][i]) + 2
            df_pdsch_demapper = _make_data_frame(myfile, desired_line_number4, 8)
            pdsch_demapper_df = pdsch_demapper_df.append(df_pdsch_demapper,
                                                         ignore_index=True, sort=False)
    
        pdsch_demapper_samp_df = pd.DataFrame()
        #    pdsch_demapper_samp_size = pdsch_demapper.shape[0]
        for i in range(0, pdsch_demapper_size, 1):
            start_line_number = int(pdsch_demapper_df["Line_Num"][i]) + 9
            samples_num = int(pdsch_demapper_df["Number of Samples"][i])
            L = int(pdsch_demapper_df["Line_Num"][i])
            demapper_samp = _get_samples(L, samples_num, start_line_number, 36, myfile)
            pdsch_demapper_samp_df = pdsch_demapper_samp_df.append(demapper_samp,
                                                                   ignore_index=True, sort=False)
        pdsch_demapper_df = pd.merge(pdsch_demapper_df, pdsch_demapper_samp_df,
                                     on="Line_Num", how="outer")
        pdsch_demapper_df = _time_extract(pdsch_demapper_df)
        # pdsch_demapper_df.to_csv('pdsch_demapper_df'+file_number+'.csv', index=True)
    
        pdcch_phich_indication_df_second = pd.DataFrame()
        pdcch_phich_indication_size = pdcch_phich_indication.shape[0]
        print("640")
        for i in range(0, pdcch_phich_indication_size, 1):
            start_line_number = int(pdcch_phich_indication["Line_Num"][i])
            df_phich = _get_ack(start_line_number, myfile)
            pdcch_phich_indication_df_second = pdcch_phich_indication_df_second.append(df_phich,
                                                                                       ignore_index=True, sort=False)
    
        pdcch_phich_indication_df = pd.DataFrame()
        pdcch_phich_indication_size = pdcch_phich_indication_df_second.shape[0]
        for i in range(0, pdcch_phich_indication_size, 1):
            desired_line_number = int(pdcch_phich_indication_df_second["Line_Num"][i]) + 5
            df_phich = _make_data_frame(myfile, desired_line_number, 0)
            pdcch_phich_indication_df = pdcch_phich_indication_df.append(df_phich, ignore_index=True, sort=False)
        pdcch_phich_indication_df = _time_extract(pdcch_phich_indication_df)
    
        pdcch_phich_indication_df_second["Line_Num"] = pdcch_phich_indication_df_second["Line_Num"].add(5)
        pdcch_phich_indication_df = pd.merge(pdcch_phich_indication_df, pdcch_phich_indication_df_second, on='Line_Num')
    
        LL1_Serving_Cell_Measurement_df = pd.DataFrame()
        LL1_Serving_Cell_Measurement_size = LL1_Serving_Cell_Measurement.shape[0]
        for i in range(0, LL1_Serving_Cell_Measurement_size, 1):
            desired_line_number = int(LL1_Serving_Cell_Measurement["Line_Num"][i]) + 2
            df_Measure = _make_data_frame(myfile, desired_line_number, 17)
            LL1_Serving_Cell_Measurement_df = LL1_Serving_Cell_Measurement_df.append(df_Measure, ignore_index=True,
                                                                                     sort=False)
        LL1_Serving_Cell_Measurement_df = _time_extract(LL1_Serving_Cell_Measurement_df)
    
        mapper_input = pdsch_demapper_df
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex='PDSCH RNTI Type').columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="Number of Tx Antennas").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="Number of Rx Antennas").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="FrequencySelectivePrecodingMatrixIndicator").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="MU Receiver Mode").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="PMI Index").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="Transmission Scheme").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="Port Enabled").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="BMOD FD Sym Index").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="MVC").columns, axis=1)  # ,"MVC Clock","MVC Request Up"
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="Traffic to Pilot Ratio").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="Carrier Index").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="CSI-RS Exists").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="ZP CSI-RS Exists").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="Log Code").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="Joint Demod Skip Reason").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="Strong ICell ID").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="Op Mode").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="PB").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="CSI-RS Symbol Skipped").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="1.25 ms counter").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="1.25 ms fraction").columns, axis=1)
        mapper_input = mapper_input.drop(mapper_input
                                         .filter(regex="PDSCH RNTI ID").columns, axis=1)
    
        # mapper_input=mapper_input.dropna(axis=1,how='all')
        #    INPUT_SIZE = mapper_input.shape[0] #1st dimension of array (number of rows)
        block_size = [col for col in mapper_input.columns if "Transport Block Size for Stream" in col]
        block_size = mapper_input[block_size]
        block_size = block_size.fillna('0')
        block_size = block_size.astype(int)
        block_size.loc[:, 'Total_Block_Size'] = block_size.sum(axis=1)  # sum all cols
        Total_Block_Size = block_size.loc[:, ['Total_Block_Size']]  # only total col
        pdsch_demapper_df = pdsch_demapper_df.join(Total_Block_Size)  # adding total col to input file
        # detect start
        for i in range(0, pdsch_demapper_df.shape[0], 1):
            count = 0
            for k in range(0, 15, 1):
                if int(pdsch_demapper_df["Number of Samples"][i + k]) >= 30:
                    count = k
                else:
                    break
            if count == 14:
                break
        start = i
        # detect end
    
        for j in range(start, pdsch_demapper_df.shape[0], 1):
            count = 0
            for k in range(0, 10, 1):
                if j + k < pdsch_demapper_df.shape[0]:
                    if int(pdsch_demapper_df["Number of Samples"][j + k]) <= 30:
                        count = k
                    else:
                        break
                else:
                    break
            if count == 9:
                break
        end = j - 1
    
        pdsch_demapper_active_time = pdsch_demapper_df.iloc[start: end].reset_index()
    
        pdsch_demapper_active_time.to_csv(os.path.dirname(thefilepath)+'PDSCH_Demapper_Active_Time' + os.path.basename(thefilepath) + '.csv', index=True)
        
        session_start = pdsch_demapper_active_time["HW Timestamp"][0]
    
        session_end = pdsch_demapper_active_time["HW Timestamp"][pdsch_demapper_active_time.shape[0] - 1]
    
        serving_cell_measurement_active_time = _active_extract(session_start, session_end,
                                                               serving_cell_measurement_df)
        serving_cell_measurement_active_time.to_csv(os.path.dirname(thefilepath)+'Serving_Cell_Measurement_Active_Time' +os.path.basename( thefilepath) + '.csv',
                                                    index=True)
    
        intra_frequency_measurement_active_time = _active_extract(session_start, session_end,
                                                                  intra_frequency_measurement_df)
        intra_frequency_measurement_active_time.to_csv(os.path.dirname(thefilepath)+'Intra_Frequency_Measurement_Active_Time' + os.path.basename( thefilepath)+ '.csv',
                                                       index=True)
    
    
        ll1_pusch_csf_log_active_time = _active_extract(session_start, session_end, ll1_pusch_csf_log_df)
        ll1_pusch_csf_log_active_time.to_csv(os.path.dirname(thefilepath)+'LL1_PUSCH_CSF_Log_Active_Time' + os.path.basename( thefilepath) + '.csv', index=True)
        pdsch_phich_indication_active_time = _active_extract(session_start - 300, session_end + 40,
                                                             pdcch_phich_indication_df)
        pdsch_phich_indication_active_time.to_csv(os.path.dirname(thefilepath)+'PDCCH_PHICH_Indication_Active_Time' + os.path.basename( thefilepath)+ '.csv',
                                                  index=True)
    
        LL1_Serving_Cell_Measurement_df = _active_extract(session_start, session_end, LL1_Serving_Cell_Measurement_df)
        LL1_Serving_Cell_Measurement_df.to_csv(os.path.dirname(thefilepath)+'LL1_Serving_Cell_Measurement' + os.path.basename( thefilepath) + '.csv', index=True)
        active_data, active_intra, active_serving, active_log, active_phich, active_LL1 = _files(thefilepath)
        summary_reportt = _summary(active_data, active_intra, active_serving, active_log, active_phich, active_LL1,
                                   thefilepath)
        # os.remove('PDSCH_Demapper_Active_Time' + thefilepath + '.csv')
        # os.remove('Serving_Cell_Measurement_Active_Time' + thefilepath + '.csv')
        # os.remove('Intra_Frequency_Measurement_Active_Time' + thefilepath + '.csv')
        # os.remove('LL1_PUSCH_CSF_Log_Active_Time' + thefilepath + '.csv')
        # os.remove('PDCCH_PHICH_Indication_Active_Time' + thefilepath + '.csv')
        # os.remove('LL1_Serving_Cell_Measurement' + thefilepath + '.csv')
    
        if path.exists("4G_data_anlysis_results.csv"):
            sumarry = pd.read_csv("4G_data_anlysis_results.csv")
            final_summary_report = pd.concat([sumarry, summary_reportt], sort=False)
            final_summary_report = final_summary_report.loc[:, ~final_summary_report.columns.str.contains('^Unnamed')]
            final_summary_report = final_summary_report.reset_index(level=0, drop=True)
            final_summary_report.to_csv('4G_data_anlysis_results.csv', index=True)
        else:
            summary_reportt = summary_reportt.loc[:, ~summary_reportt.columns.str.contains('^Unnamed')]
            summary_reportt = summary_reportt.reset_index(level=0, drop=True)
            summary_reportt.to_csv('4G_data_anlysis_results.csv', index=True)
        summary_reportt = pd.read_csv("4G_data_anlysis_results.csv")
        os.remove('4G_data_anlysis_results.csv')
        print("analysis_finished")
    
        return summary_reportt
    ########################################################################################################################
    def _Model(thefilepath, Train_path):
        summary_reportt = _Analysis_(thefilepath)
        dataset = pd.read_csv(Train_path)
        dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
        TEST = summary_reportt[["Acked Throughput[kbit/s]","Acked Transport block size mean","File","File duration[sec]","Inst RSRQ mean [dB]","Inst SINR RX0 mean [dB]","Inst SINR_RX1 mean [dB]","Mobile rank mean","Modulation bits mean","Network rank mean","No of RBs [bits]","Scheduling_percentage[%]","Serving Filtered RSRP mean[dBm]", "BW","WideBand CQI_CW0 mean"]]
        TEST["Scheduling_percentage[%]"] = TEST["Scheduling_percentage[%]"].replace(regex=["%"], value='').astype(
            'float')
        
        dataset["Scheduling_percentage[%]"] = dataset["Scheduling_percentage[%]"].replace(regex=["%"], value='').astype(
            'float')
        
        #Regression
        y_Reg = dataset["Acked Throughput[kbit/s]"].values
        X_Reg = dataset.iloc[:, 5:-1].values
        
        y_test = TEST["Acked Throughput[kbit/s]"].values
        X_test = TEST.iloc[:, 5:].values
        
        y_Reg = np.reshape( np.array(y_Reg), (len(y_Reg), 1))
        y_test = np.reshape( np.array(y_test), (len(y_test), 1))
        
        # Fitting Random Forest Regression to the dataset
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
        regressor.fit(X_Reg, y_Reg)
        
        # Predicting a new result
        y_pred_Reg = regressor.predict(X_test)
        y_pred_Reg = np.reshape( np.array(y_pred_Reg), (len(y_pred_Reg), 1))
        
        
        Deviation = np.subtract(y_pred_Reg,y_test)
        Deviation = pd.DataFrame(Deviation,columns=["Deviation[kbit/s]"])
        
        Regression_df = pd.DataFrame(y_pred_Reg,columns=["Predicted Throughput[kbit/s]"])
        
        df = Regression_df.join(Deviation) 
        TestDataset = df.join(TEST) 
        TestDataset = TestDataset.loc[:, ~TestDataset.columns.str.contains('^Unnamed')]
    
    #        TestDataset.drop(TestDataset.columns[TestDataset.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        
        #classifications
        y_class = dataset["State"].values
        X_class = dataset.iloc[:, 5:-1].values + np.reshape( np.array(dataset["Acked Throughput[kbit/s]"].values), (len(dataset["Acked Throughput[kbit/s]"].values), 1))
        
        
        X_test = TEST.iloc[:, 5:].values + np.reshape( np.array(TEST["Acked Throughput[kbit/s]"].values), (len(TEST["Acked Throughput[kbit/s]"].values), 1))
        
        y_class = np.reshape( np.array(y_class), (len(y_class), 1))            
        # Fitting Random Forest Classification to the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
        classifier.fit(X_class, y_class)
        
        # Predicting the Test set results
        y_pred_class = classifier.predict(X_test)
        
                
        y_pred_class = np.reshape( np.array(y_pred_class), (len(y_pred_class), 1))
        classifications_df = pd.DataFrame(y_pred_class,columns=["Predicted State"])
        TestDataset = TestDataset.join(classifications_df)
        TestDataset = TestDataset.loc[:, ~TestDataset.columns.str.contains('^Unnamed')] 
        TestDataset.to_csv("TestDataset.csv", index=True)
        TestDataset.to_csv("xx.csv", index=True)
###############################################################################
    ELAPSED1 = time.time()
    # insert the file name inside main

    thefilepath = Data1
    Train_path = TRAIN
    _Model(thefilepath, Train_path)
    ELAPSED2 = time.time()
    print("elapsed:", ELAPSED2 - ELAPSED1)

X = LTE('File10_textexp.txt', 'TrainDatastate.csv')

