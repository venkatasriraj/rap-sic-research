#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: venka
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import blocks
from gnuradio import blocks, gr
from gnuradio import channels
from gnuradio.filter import firdes
from gnuradio import digital
from gnuradio import filter
import accessCode_epy_block_0_0 as epy_block_0_0  # embedded python block
import accessCode_epy_block_0_0_0 as epy_block_0_0_0  # embedded python block
import accessCode_epy_block_0_0_0_0 as epy_block_0_0_0_0  # embedded python block
import accessCode_epy_block_0_0_0_0_0 as epy_block_0_0_0_0_0  # embedded python block
import accessCode_epy_block_0_1 as epy_block_0_1  # embedded python block
import accessCode_epy_block_2_0 as epy_block_2_0  # embedded python block
import accessCode_epy_block_2_1 as epy_block_2_1  # embedded python block
import accessCode_epy_block_2_1_0 as epy_block_2_1_0  # embedded python block
import accessCode_epy_block_2_1_1 as epy_block_2_1_1  # embedded python block
import accessCode_epy_block_2_1_1_0 as epy_block_2_1_1_0  # embedded python block
import accessCode_epy_block_2_1_1_0_0 as epy_block_2_1_1_0_0  # embedded python block
import accessCode_epy_block_3 as epy_block_3  # embedded python block
import sip
import threading
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation




class accessCode(gr.top_block, Qt.QWidget):

    def __init__(self, iq='iqRx.csv', log_file1='data_tx1.csv', log_file2='data_tx2.csv', log_file3='data_tx3.csv', log_file4='data_tx4.csv', log_file5='data_tx5.csv', rx_log='rx_log.csv'):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "accessCode")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Parameters
        ##################################################
        self.iq = iq
        self.log_file1 = log_file1
        self.log_file2 = log_file2
        self.log_file3 = log_file3
        self.log_file4 = log_file4
        self.log_file5 = log_file5
        self.rx_log = rx_log

        ##################################################
        # Variables
        ##################################################
        self.sps = sps = 10
        self.samp_rate = samp_rate = 40000
        self.excess_bw = excess_bw = 0.35
        self.bw = bw = (1+excess_bw)*(samp_rate//sps)
        self.usrp_rate = usrp_rate = 40e3
        self.user_id5 = user_id5 = 5
        self.user_id4 = user_id4 = 4
        self.user_id3 = user_id3 = 3
        self.user_id2 = user_id2 = 2
        self.user_id1 = user_id1 = 1
        self.time_offset = time_offset = 1.000
        self.thresh = thresh = 32
        self.taps = taps = [1.0 + 0.0j, ]
        self.rs_ratio = rs_ratio = 1.0
        self.phase_bw = phase_bw = 0.0628
        self.packet_size = packet_size = 80
        self.noise_volt = noise_volt = 0.0
        self.low_pass_filter_taps = low_pass_filter_taps = firdes.low_pass(1.0, samp_rate, (bw//2)+5e3, 10e3, window.WIN_HAMMING, 6.76)
        self.freq_offset = freq_offset = 0
        self.bpsk = bpsk = digital.constellation_bpsk().base()
        self.bpsk.set_npwr(1.0)

        ##################################################
        # Blocks
        ##################################################

        self._time_offset_range = qtgui.Range(0.999, 1.001, 0.0001, 1.000, 200)
        self._time_offset_win = qtgui.RangeWidget(self._time_offset_range, self.set_time_offset, "Timing Offset", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._time_offset_win)
        self._noise_volt_range = qtgui.Range(0, 1, 0.01, 0.0, 200)
        self._noise_volt_win = qtgui.RangeWidget(self._noise_volt_range, self.set_noise_volt, "Noise Voltage", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._noise_volt_win)
        self._freq_offset_range = qtgui.Range(-0.1, 0.1, 0.001, 0, 200)
        self._freq_offset_win = qtgui.RangeWidget(self._freq_offset_range, self.set_freq_offset, "Frequency Offset", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._freq_offset_win)
        self.root_raised_cosine_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.root_raised_cosine(
                1,
                samp_rate,
                ((samp_rate//sps)),
                0.35,
                (11*sps)))
        self.qtgui_time_sink_x_0_2 = qtgui.time_sink_f(
            2048, #size
            samp_rate, #samp_rate
            'Correlate input', #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0_2.set_update_time(0.10)
        self.qtgui_time_sink_x_0_2.set_y_axis(-0.1, 1.1)

        self.qtgui_time_sink_x_0_2.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0_2.enable_tags(True)
        self.qtgui_time_sink_x_0_2.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.2, 0.0, 0, "packet_len")
        self.qtgui_time_sink_x_0_2.enable_autoscale(True)
        self.qtgui_time_sink_x_0_2.enable_grid(True)
        self.qtgui_time_sink_x_0_2.enable_axis_labels(True)
        self.qtgui_time_sink_x_0_2.enable_control_panel(False)
        self.qtgui_time_sink_x_0_2.enable_stem_plot(False)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_0_2.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_0_2.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0_2.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0_2.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0_2.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0_2.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0_2.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_2_win = sip.wrapinstance(self.qtgui_time_sink_x_0_2.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_time_sink_x_0_2_win, 1, 0, 1, 2)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_freq_sink_x_0_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            'RX_Sym', #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0_0.enable_grid(False)
        self.qtgui_freq_sink_x_0_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_x_0_0_win, 0, 0, 1, 2)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_const_sink_x_0_0 = qtgui.const_sink_c(
            1024, #size
            'RX_Cos', #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0_0.set_y_axis((-2), 2)
        self.qtgui_const_sink_x_0_0.set_x_axis((-2), 2)
        self.qtgui_const_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0.enable_grid(False)
        self.qtgui_const_sink_x_0_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_const_sink_x_0_0_win, 0, 2, 1, 2)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.fft_filter_xxx_0_0_0_0_1_0_0 = filter.fft_filter_ccc(1, low_pass_filter_taps, 1)
        self.fft_filter_xxx_0_0_0_0_1_0_0.declare_sample_delay(0)
        self.fft_filter_xxx_0_0_0_0_1_0 = filter.fft_filter_ccc(1, low_pass_filter_taps, 1)
        self.fft_filter_xxx_0_0_0_0_1_0.declare_sample_delay(0)
        self.fft_filter_xxx_0_0_0_0_1 = filter.fft_filter_ccc(1, low_pass_filter_taps, 1)
        self.fft_filter_xxx_0_0_0_0_1.declare_sample_delay(0)
        self.fft_filter_xxx_0_0_0_0_0 = filter.fft_filter_ccc(1, low_pass_filter_taps, 1)
        self.fft_filter_xxx_0_0_0_0_0.declare_sample_delay(0)
        self.fft_filter_xxx_0_0_0_0 = filter.fft_filter_ccc(1, low_pass_filter_taps, 1)
        self.fft_filter_xxx_0_0_0_0.declare_sample_delay(0)
        self.epy_block_3 = epy_block_3.access_code_correlator(packet_size=packet_size, threshold=thresh, log_file=rx_log)
        self.epy_block_2_1_1_0_0 = epy_block_2_1_1_0_0.Random_Packet_Generator(mean_interval=1, packet_size=packet_size, user_id=user_id5, log_file=log_file5, total_packets=70, initial_delay=0)
        self.epy_block_2_1_1_0 = epy_block_2_1_1_0.Random_Packet_Generator(mean_interval=1, packet_size=packet_size, user_id=user_id4, log_file=log_file4, total_packets=70, initial_delay=0)
        self.epy_block_2_1_1 = epy_block_2_1_1.Random_Packet_Generator(mean_interval=1, packet_size=packet_size, user_id=user_id3, log_file=log_file3, total_packets=70, initial_delay=0)
        self.epy_block_2_1_0 = epy_block_2_1_0.Random_Packet_Generator(mean_interval=1, packet_size=packet_size, user_id=user_id1, log_file=log_file1, total_packets=70, initial_delay=0)
        self.epy_block_2_1 = epy_block_2_1.Random_Packet_Generator(mean_interval=1, packet_size=packet_size, user_id=user_id2, log_file=log_file2, total_packets=70, initial_delay=0)
        self.epy_block_2_0 = epy_block_2_0.iq_logger_with_timestamp(iq_csv_filename=iq)
        self.epy_block_0_1 = epy_block_0_1.PDU_to_Timed_Byte_Stream()
        self.epy_block_0_0_0_0_0 = epy_block_0_0_0_0_0.PDU_to_Timed_Byte_Stream()
        self.epy_block_0_0_0_0 = epy_block_0_0_0_0.PDU_to_Timed_Byte_Stream()
        self.epy_block_0_0_0 = epy_block_0_0_0.PDU_to_Timed_Byte_Stream()
        self.epy_block_0_0 = epy_block_0_0.PDU_to_Timed_Byte_Stream()
        self.digital_symbol_sync_xx_0 = digital.symbol_sync_cc(
            digital.TED_MUELLER_AND_MULLER,
            sps,
            phase_bw,
            1.0,
            1.0,
            1.5,
            1,
            digital.constellation_bpsk().base(),
            digital.IR_MMSE_8TAP,
            128,
            [])
        self.digital_map_bb_0 = digital.map_bb([0,1])
        self.digital_fll_band_edge_cc_0 = digital.fll_band_edge_cc(sps, excess_bw, 44, phase_bw, False)
        self.digital_diff_decoder_bb_0 = digital.diff_decoder_bb(2, digital.DIFF_DIFFERENTIAL)
        self.digital_costas_loop_cc_0 = digital.costas_loop_cc(phase_bw, 2, False)
        self.digital_constellation_modulator_0_0_0_1_0_0 = digital.generic_mod(
            constellation=bpsk,
            differential=True,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=excess_bw,
            verbose=False,
            log=False,
            truncate=False)
        self.digital_constellation_modulator_0_0_0_1_0 = digital.generic_mod(
            constellation=bpsk,
            differential=True,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=excess_bw,
            verbose=False,
            log=False,
            truncate=False)
        self.digital_constellation_modulator_0_0_0_1 = digital.generic_mod(
            constellation=bpsk,
            differential=True,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=excess_bw,
            verbose=False,
            log=False,
            truncate=False)
        self.digital_constellation_modulator_0_0_0_0 = digital.generic_mod(
            constellation=bpsk,
            differential=True,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=excess_bw,
            verbose=False,
            log=False,
            truncate=False)
        self.digital_constellation_modulator_0_0_0 = digital.generic_mod(
            constellation=bpsk,
            differential=True,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=excess_bw,
            verbose=False,
            log=False,
            truncate=False)
        self.digital_constellation_decoder_cb_0 = digital.constellation_decoder_cb(bpsk)
        self.channels_channel_model_0 = channels.channel_model(
            noise_voltage=noise_volt,
            frequency_offset=freq_offset,
            epsilon=time_offset,
            taps=taps,
            noise_seed=0,
            block_tags=True)
        self.blocks_uchar_to_float_0_1 = blocks.uchar_to_float()
        self.blocks_uchar_to_float_0_0_1_0_0 = blocks.uchar_to_float()
        self.blocks_uchar_to_float_0_0_1_0 = blocks.uchar_to_float()
        self.blocks_uchar_to_float_0_0_1 = blocks.uchar_to_float()
        self.blocks_uchar_to_float_0_0_0 = blocks.uchar_to_float()
        self.blocks_uchar_to_float_0_0 = blocks.uchar_to_float()
        self.blocks_throttle2_0_0 = blocks.throttle( gr.sizeof_gr_complex*1, samp_rate, True, 0 if "auto" == "auto" else max( int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1) )
        self.blocks_repeat_0_0_1_0_0 = blocks.repeat(gr.sizeof_float*1, (8*sps))
        self.blocks_repeat_0_0_1_0 = blocks.repeat(gr.sizeof_float*1, (8*sps))
        self.blocks_repeat_0_0_1 = blocks.repeat(gr.sizeof_float*1, (8*sps))
        self.blocks_repeat_0_0_0 = blocks.repeat(gr.sizeof_float*1, (8*sps))
        self.blocks_repeat_0_0 = blocks.repeat(gr.sizeof_float*1, (8*sps))
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_char*1)
        self.blocks_multiply_xx_0_0_1_0_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0_0_1_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0_0_1 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0_0_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0_0 = blocks.multiply_vcc(1)
        self.blocks_message_debug_0 = blocks.message_debug(True, gr.log_levels.info)
        self.blocks_float_to_complex_0_0_1_0_0 = blocks.float_to_complex(1)
        self.blocks_float_to_complex_0_0_1_0 = blocks.float_to_complex(1)
        self.blocks_float_to_complex_0_0_1 = blocks.float_to_complex(1)
        self.blocks_float_to_complex_0_0_0 = blocks.float_to_complex(1)
        self.blocks_float_to_complex_0_0 = blocks.float_to_complex(1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_const_source_x_0_0_1_0_0 = analog.sig_source_f(0, analog.GR_CONST_WAVE, 0, 0, 0)
        self.analog_const_source_x_0_0_1_0 = analog.sig_source_f(0, analog.GR_CONST_WAVE, 0, 0, 0)
        self.analog_const_source_x_0_0_1 = analog.sig_source_f(0, analog.GR_CONST_WAVE, 0, 0, 0)
        self.analog_const_source_x_0_0_0 = analog.sig_source_f(0, analog.GR_CONST_WAVE, 0, 0, 0)
        self.analog_const_source_x_0_0 = analog.sig_source_f(0, analog.GR_CONST_WAVE, 0, 0, 0)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.epy_block_2_1, 'pdu_out'), (self.epy_block_0_0, 'pdus'))
        self.msg_connect((self.epy_block_2_1_0, 'pdu_out'), (self.epy_block_0_1, 'pdus'))
        self.msg_connect((self.epy_block_2_1_1, 'pdu_out'), (self.epy_block_0_0_0, 'pdus'))
        self.msg_connect((self.epy_block_2_1_1_0, 'pdu_out'), (self.epy_block_0_0_0_0, 'pdus'))
        self.msg_connect((self.epy_block_2_1_1_0_0, 'pdu_out'), (self.epy_block_0_0_0_0_0, 'pdus'))
        self.msg_connect((self.epy_block_3, 'pdu_out'), (self.blocks_message_debug_0, 'print'))
        self.connect((self.analog_const_source_x_0_0, 0), (self.blocks_float_to_complex_0_0, 1))
        self.connect((self.analog_const_source_x_0_0_0, 0), (self.blocks_float_to_complex_0_0_0, 1))
        self.connect((self.analog_const_source_x_0_0_1, 0), (self.blocks_float_to_complex_0_0_1, 1))
        self.connect((self.analog_const_source_x_0_0_1_0, 0), (self.blocks_float_to_complex_0_0_1_0, 1))
        self.connect((self.analog_const_source_x_0_0_1_0_0, 0), (self.blocks_float_to_complex_0_0_1_0_0, 1))
        self.connect((self.blocks_add_xx_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.blocks_float_to_complex_0_0, 0), (self.blocks_multiply_xx_0_0, 1))
        self.connect((self.blocks_float_to_complex_0_0_0, 0), (self.blocks_multiply_xx_0_0_0, 1))
        self.connect((self.blocks_float_to_complex_0_0_1, 0), (self.blocks_multiply_xx_0_0_1, 1))
        self.connect((self.blocks_float_to_complex_0_0_1_0, 0), (self.blocks_multiply_xx_0_0_1_0, 1))
        self.connect((self.blocks_float_to_complex_0_0_1_0_0, 0), (self.blocks_multiply_xx_0_0_1_0_0, 1))
        self.connect((self.blocks_multiply_xx_0_0, 0), (self.fft_filter_xxx_0_0_0_0, 0))
        self.connect((self.blocks_multiply_xx_0_0_0, 0), (self.fft_filter_xxx_0_0_0_0_0, 0))
        self.connect((self.blocks_multiply_xx_0_0_1, 0), (self.fft_filter_xxx_0_0_0_0_1, 0))
        self.connect((self.blocks_multiply_xx_0_0_1_0, 0), (self.fft_filter_xxx_0_0_0_0_1_0, 0))
        self.connect((self.blocks_multiply_xx_0_0_1_0_0, 0), (self.fft_filter_xxx_0_0_0_0_1_0_0, 0))
        self.connect((self.blocks_repeat_0_0, 0), (self.blocks_float_to_complex_0_0, 0))
        self.connect((self.blocks_repeat_0_0_0, 0), (self.blocks_float_to_complex_0_0_0, 0))
        self.connect((self.blocks_repeat_0_0_1, 0), (self.blocks_float_to_complex_0_0_1, 0))
        self.connect((self.blocks_repeat_0_0_1_0, 0), (self.blocks_float_to_complex_0_0_1_0, 0))
        self.connect((self.blocks_repeat_0_0_1_0_0, 0), (self.blocks_float_to_complex_0_0_1_0_0, 0))
        self.connect((self.blocks_throttle2_0_0, 0), (self.root_raised_cosine_filter_0, 0))
        self.connect((self.blocks_uchar_to_float_0_0, 0), (self.blocks_repeat_0_0, 0))
        self.connect((self.blocks_uchar_to_float_0_0_0, 0), (self.qtgui_time_sink_x_0_2, 0))
        self.connect((self.blocks_uchar_to_float_0_0_1, 0), (self.blocks_repeat_0_0_1, 0))
        self.connect((self.blocks_uchar_to_float_0_0_1_0, 0), (self.blocks_repeat_0_0_1_0, 0))
        self.connect((self.blocks_uchar_to_float_0_0_1_0_0, 0), (self.blocks_repeat_0_0_1_0_0, 0))
        self.connect((self.blocks_uchar_to_float_0_1, 0), (self.blocks_repeat_0_0_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_throttle2_0_0, 0))
        self.connect((self.digital_constellation_decoder_cb_0, 0), (self.digital_diff_decoder_bb_0, 0))
        self.connect((self.digital_constellation_modulator_0_0_0, 0), (self.blocks_multiply_xx_0_0, 0))
        self.connect((self.digital_constellation_modulator_0_0_0_0, 0), (self.blocks_multiply_xx_0_0_0, 0))
        self.connect((self.digital_constellation_modulator_0_0_0_1, 0), (self.blocks_multiply_xx_0_0_1, 0))
        self.connect((self.digital_constellation_modulator_0_0_0_1_0, 0), (self.blocks_multiply_xx_0_0_1_0, 0))
        self.connect((self.digital_constellation_modulator_0_0_0_1_0_0, 0), (self.blocks_multiply_xx_0_0_1_0_0, 0))
        self.connect((self.digital_costas_loop_cc_0, 0), (self.epy_block_2_0, 0))
        self.connect((self.digital_costas_loop_cc_0, 0), (self.qtgui_const_sink_x_0_0, 0))
        self.connect((self.digital_diff_decoder_bb_0, 0), (self.digital_map_bb_0, 0))
        self.connect((self.digital_fll_band_edge_cc_0, 0), (self.digital_symbol_sync_xx_0, 0))
        self.connect((self.digital_fll_band_edge_cc_0, 0), (self.qtgui_freq_sink_x_0_0, 0))
        self.connect((self.digital_map_bb_0, 0), (self.blocks_uchar_to_float_0_0_0, 0))
        self.connect((self.digital_map_bb_0, 0), (self.epy_block_3, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.digital_costas_loop_cc_0, 0))
        self.connect((self.epy_block_0_0, 1), (self.blocks_uchar_to_float_0_0, 0))
        self.connect((self.epy_block_0_0, 0), (self.digital_constellation_modulator_0_0_0, 0))
        self.connect((self.epy_block_0_0_0, 1), (self.blocks_uchar_to_float_0_0_1, 0))
        self.connect((self.epy_block_0_0_0, 0), (self.digital_constellation_modulator_0_0_0_1, 0))
        self.connect((self.epy_block_0_0_0_0, 1), (self.blocks_uchar_to_float_0_0_1_0, 0))
        self.connect((self.epy_block_0_0_0_0, 0), (self.digital_constellation_modulator_0_0_0_1_0, 0))
        self.connect((self.epy_block_0_0_0_0_0, 1), (self.blocks_uchar_to_float_0_0_1_0_0, 0))
        self.connect((self.epy_block_0_0_0_0_0, 0), (self.digital_constellation_modulator_0_0_0_1_0_0, 0))
        self.connect((self.epy_block_0_1, 1), (self.blocks_uchar_to_float_0_1, 0))
        self.connect((self.epy_block_0_1, 0), (self.digital_constellation_modulator_0_0_0_0, 0))
        self.connect((self.epy_block_2_0, 0), (self.digital_constellation_decoder_cb_0, 0))
        self.connect((self.epy_block_3, 0), (self.blocks_null_sink_0, 0))
        self.connect((self.fft_filter_xxx_0_0_0_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.fft_filter_xxx_0_0_0_0_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.fft_filter_xxx_0_0_0_0_1, 0), (self.blocks_add_xx_0, 2))
        self.connect((self.fft_filter_xxx_0_0_0_0_1_0, 0), (self.blocks_add_xx_0, 3))
        self.connect((self.fft_filter_xxx_0_0_0_0_1_0_0, 0), (self.blocks_add_xx_0, 4))
        self.connect((self.root_raised_cosine_filter_0, 0), (self.digital_fll_band_edge_cc_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "accessCode")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_iq(self):
        return self.iq

    def set_iq(self, iq):
        self.iq = iq
        self.epy_block_2_0.iq_csv_filename = self.iq

    def get_log_file1(self):
        return self.log_file1

    def set_log_file1(self, log_file1):
        self.log_file1 = log_file1
        self.epy_block_2_1_0.log_file = self.log_file1

    def get_log_file2(self):
        return self.log_file2

    def set_log_file2(self, log_file2):
        self.log_file2 = log_file2
        self.epy_block_2_1.log_file = self.log_file2

    def get_log_file3(self):
        return self.log_file3

    def set_log_file3(self, log_file3):
        self.log_file3 = log_file3
        self.epy_block_2_1_1.log_file = self.log_file3

    def get_log_file4(self):
        return self.log_file4

    def set_log_file4(self, log_file4):
        self.log_file4 = log_file4
        self.epy_block_2_1_1_0.log_file = self.log_file4

    def get_log_file5(self):
        return self.log_file5

    def set_log_file5(self, log_file5):
        self.log_file5 = log_file5
        self.epy_block_2_1_1_0_0.log_file = self.log_file5

    def get_rx_log(self):
        return self.rx_log

    def set_rx_log(self, rx_log):
        self.rx_log = rx_log
        self.epy_block_3.log_file = self.rx_log

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_bw((1+self.excess_bw)*(self.samp_rate//self.sps))
        self.blocks_repeat_0_0.set_interpolation((8*self.sps))
        self.blocks_repeat_0_0_0.set_interpolation((8*self.sps))
        self.blocks_repeat_0_0_1.set_interpolation((8*self.sps))
        self.blocks_repeat_0_0_1_0.set_interpolation((8*self.sps))
        self.blocks_repeat_0_0_1_0_0.set_interpolation((8*self.sps))
        self.digital_symbol_sync_xx_0.set_sps(self.sps)
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, ((self.samp_rate//self.sps)), 0.35, (11*self.sps)))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_bw((1+self.excess_bw)*(self.samp_rate//self.sps))
        self.set_low_pass_filter_taps(firdes.low_pass(1.0, self.samp_rate, (self.bw//2)+5e3, 10e3, window.WIN_HAMMING, 6.76))
        self.blocks_throttle2_0_0.set_sample_rate(self.samp_rate)
        self.qtgui_freq_sink_x_0_0.set_frequency_range(0, self.samp_rate)
        self.qtgui_time_sink_x_0_2.set_samp_rate(self.samp_rate)
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, ((self.samp_rate//self.sps)), 0.35, (11*self.sps)))

    def get_excess_bw(self):
        return self.excess_bw

    def set_excess_bw(self, excess_bw):
        self.excess_bw = excess_bw
        self.set_bw((1+self.excess_bw)*(self.samp_rate//self.sps))

    def get_bw(self):
        return self.bw

    def set_bw(self, bw):
        self.bw = bw
        self.set_low_pass_filter_taps(firdes.low_pass(1.0, self.samp_rate, (self.bw//2)+5e3, 10e3, window.WIN_HAMMING, 6.76))

    def get_usrp_rate(self):
        return self.usrp_rate

    def set_usrp_rate(self, usrp_rate):
        self.usrp_rate = usrp_rate

    def get_user_id5(self):
        return self.user_id5

    def set_user_id5(self, user_id5):
        self.user_id5 = user_id5
        self.epy_block_2_1_1_0_0.user_id = self.user_id5

    def get_user_id4(self):
        return self.user_id4

    def set_user_id4(self, user_id4):
        self.user_id4 = user_id4
        self.epy_block_2_1_1_0.user_id = self.user_id4

    def get_user_id3(self):
        return self.user_id3

    def set_user_id3(self, user_id3):
        self.user_id3 = user_id3
        self.epy_block_2_1_1.user_id = self.user_id3

    def get_user_id2(self):
        return self.user_id2

    def set_user_id2(self, user_id2):
        self.user_id2 = user_id2
        self.epy_block_2_1.user_id = self.user_id2

    def get_user_id1(self):
        return self.user_id1

    def set_user_id1(self, user_id1):
        self.user_id1 = user_id1
        self.epy_block_2_1_0.user_id = self.user_id1

    def get_time_offset(self):
        return self.time_offset

    def set_time_offset(self, time_offset):
        self.time_offset = time_offset
        self.channels_channel_model_0.set_timing_offset(self.time_offset)

    def get_thresh(self):
        return self.thresh

    def set_thresh(self, thresh):
        self.thresh = thresh
        self.epy_block_3.threshold = self.thresh

    def get_taps(self):
        return self.taps

    def set_taps(self, taps):
        self.taps = taps
        self.channels_channel_model_0.set_taps(self.taps)

    def get_rs_ratio(self):
        return self.rs_ratio

    def set_rs_ratio(self, rs_ratio):
        self.rs_ratio = rs_ratio

    def get_phase_bw(self):
        return self.phase_bw

    def set_phase_bw(self, phase_bw):
        self.phase_bw = phase_bw
        self.digital_costas_loop_cc_0.set_loop_bandwidth(self.phase_bw)
        self.digital_fll_band_edge_cc_0.set_loop_bandwidth(self.phase_bw)
        self.digital_symbol_sync_xx_0.set_loop_bandwidth(self.phase_bw)

    def get_packet_size(self):
        return self.packet_size

    def set_packet_size(self, packet_size):
        self.packet_size = packet_size
        self.epy_block_2_1.packet_size = self.packet_size
        self.epy_block_2_1_0.packet_size = self.packet_size
        self.epy_block_2_1_1.packet_size = self.packet_size
        self.epy_block_2_1_1_0.packet_size = self.packet_size
        self.epy_block_2_1_1_0_0.packet_size = self.packet_size
        self.epy_block_3.packet_size = self.packet_size

    def get_noise_volt(self):
        return self.noise_volt

    def set_noise_volt(self, noise_volt):
        self.noise_volt = noise_volt
        self.channels_channel_model_0.set_noise_voltage(self.noise_volt)

    def get_low_pass_filter_taps(self):
        return self.low_pass_filter_taps

    def set_low_pass_filter_taps(self, low_pass_filter_taps):
        self.low_pass_filter_taps = low_pass_filter_taps
        self.fft_filter_xxx_0_0_0_0.set_taps(self.low_pass_filter_taps)
        self.fft_filter_xxx_0_0_0_0_0.set_taps(self.low_pass_filter_taps)
        self.fft_filter_xxx_0_0_0_0_1.set_taps(self.low_pass_filter_taps)
        self.fft_filter_xxx_0_0_0_0_1_0.set_taps(self.low_pass_filter_taps)
        self.fft_filter_xxx_0_0_0_0_1_0_0.set_taps(self.low_pass_filter_taps)

    def get_freq_offset(self):
        return self.freq_offset

    def set_freq_offset(self, freq_offset):
        self.freq_offset = freq_offset
        self.channels_channel_model_0.set_frequency_offset(self.freq_offset)

    def get_bpsk(self):
        return self.bpsk

    def set_bpsk(self, bpsk):
        self.bpsk = bpsk
        self.digital_constellation_decoder_cb_0.set_constellation(self.bpsk)



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--iq", dest="iq", type=str, default='iqRx.csv',
        help="Set filename [default=%(default)r]")
    parser.add_argument(
        "--log-file1", dest="log_file1", type=str, default='data_tx1.csv',
        help="Set File Name [default=%(default)r]")
    parser.add_argument(
        "--log-file2", dest="log_file2", type=str, default='data_tx2.csv',
        help="Set File Name [default=%(default)r]")
    parser.add_argument(
        "--log-file3", dest="log_file3", type=str, default='data_tx3.csv',
        help="Set File Name [default=%(default)r]")
    parser.add_argument(
        "--log-file4", dest="log_file4", type=str, default='data_tx4.csv',
        help="Set File Name [default=%(default)r]")
    parser.add_argument(
        "--log-file5", dest="log_file5", type=str, default='data_tx5.csv',
        help="Set File Name [default=%(default)r]")
    parser.add_argument(
        "--rx-log", dest="rx_log", type=str, default='rx_log.csv',
        help="Set file [default=%(default)r]")
    return parser


def main(top_block_cls=accessCode, options=None):
    if options is None:
        options = argument_parser().parse_args()

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(iq=options.iq, log_file1=options.log_file1, log_file2=options.log_file2, log_file3=options.log_file3, log_file4=options.log_file4, log_file5=options.log_file5, rx_log=options.rx_log)

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
