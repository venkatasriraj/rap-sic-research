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
from gnuradio import analog
from gnuradio import blocks
import pmt
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
import TX_c_epy_block_0 as epy_block_0  # embedded python block
import TX_c_epy_block_0_0 as epy_block_0_0  # embedded python block
import TX_c_epy_block_0_1 as epy_block_0_1  # embedded python block
import TX_c_epy_block_0_1_0 as epy_block_0_1_0  # embedded python block
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




class TX_c(gr.top_block, Qt.QWidget):

    def __init__(self):
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "TX_c")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.sps = sps = 5
        self.samp_rate = samp_rate = 40e3
        self.excess_bw = excess_bw = 0.35
        self.packet_size = packet_size = 100
        self.bw = bw = (1+excess_bw)*(samp_rate//sps)
        self.bps = bps = 1
        self.usrp_rate = usrp_rate = 40e3
        self.user_id2 = user_id2 = 2
        self.user_id1 = user_id1 = 1
        self.thresh = thresh = 31
        self.taps = taps = [1.0 + 0.0j, ]
        self.rs_ratio = rs_ratio = 1.0
        self.phase_bw = phase_bw = 0.0628
        self.low_pass_filter_taps = low_pass_filter_taps = firdes.low_pass(1.0, samp_rate, (bw//2)+5e3, 10e3, window.WIN_HAMMING, 6.76)
        self.initial_delay = initial_delay = 2
        self.dist2 = dist2 = [(2, 1.00)]
        self.dist1 = dist1 = [(1, 1.00)]
        self.bpsk = bpsk = digital.constellation_bpsk().base()
        self.bpsk.set_npwr(1.0)
        self.T_slot = T_slot = ((packet_size)*8*sps)/(bps*samp_rate)
        self.N_Slots = N_Slots = 2

        ##################################################
        # Blocks
        ##################################################

        self.root_raised_cosine_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.root_raised_cosine(
                1,
                samp_rate,
                ((samp_rate//sps)),
                0.35,
                (11*sps)))
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
        self.fft_filter_xxx_0_0_0_0_0_0_0 = filter.fft_filter_ccc(1, low_pass_filter_taps, 1)
        self.fft_filter_xxx_0_0_0_0_0_0_0.declare_sample_delay(0)
        self.fft_filter_xxx_0_0_0_0_0_0 = filter.fft_filter_ccc(1, low_pass_filter_taps, 1)
        self.fft_filter_xxx_0_0_0_0_0_0.declare_sample_delay(0)
        self.fft_filter_xxx_0_0_0_0_0 = filter.fft_filter_ccc(1, low_pass_filter_taps, 1)
        self.fft_filter_xxx_0_0_0_0_0.declare_sample_delay(0)
        self.epy_block_0_1_0 = epy_block_0_1_0.PDU_to_Timed_Byte_Stream()
        self.epy_block_0_1 = epy_block_0_1.PDU_to_Timed_Byte_Stream()
        self.epy_block_0_0 = epy_block_0_0.IRSA_Packet_Generator(N_slots=N_Slots, slot_duration=T_slot, lambda_dist=dist2, user_id=user_id2, packet_size=packet_size, total_packets=1, tx_probability=1, initial_delay=initial_delay)
        self.epy_block_0 = epy_block_0.IRSA_Packet_Generator(N_slots=N_Slots, slot_duration=T_slot, lambda_dist=dist1, user_id=user_id1, packet_size=packet_size, total_packets=1, tx_probability=1, initial_delay=initial_delay)
        self.digital_constellation_modulator_0_0_0_0_0 = digital.generic_mod(
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
        self.blocks_uchar_to_float_0_1_0 = blocks.uchar_to_float()
        self.blocks_uchar_to_float_0_1 = blocks.uchar_to_float()
        self.blocks_throttle2_0_0 = blocks.throttle( gr.sizeof_gr_complex*1, samp_rate, True, 0 if "auto" == "auto" else max( int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1) )
        self.blocks_repeat_0_0_0_0 = blocks.repeat(gr.sizeof_float*1, (8*sps))
        self.blocks_repeat_0_0_0 = blocks.repeat(gr.sizeof_float*1, (8*sps))
        self.blocks_null_sink_1 = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.blocks_multiply_xx_0_0_0_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0_0_0 = blocks.multiply_vcc(1)
        self.blocks_message_strobe_0 = blocks.message_strobe(pmt.intern("TEST"), ((int)(N_Slots * T_slot * 1e3)))
        self.blocks_float_to_complex_0_0_0_0 = blocks.float_to_complex(1)
        self.blocks_float_to_complex_0_0_0 = blocks.float_to_complex(1)
        self.blocks_file_sink_0_0 = blocks.file_sink(gr.sizeof_gr_complex*1, 'rc_samples.bin', False)
        self.blocks_file_sink_0_0.set_unbuffered(False)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_const_source_x_0_0_0_0 = analog.sig_source_f(0, analog.GR_CONST_WAVE, 0, 0, 0)
        self.analog_const_source_x_0_0_0 = analog.sig_source_f(0, analog.GR_CONST_WAVE, 0, 0, 0)
        self.analog_const_source_x_0 = analog.sig_source_c(0, analog.GR_CONST_WAVE, 0, 0, 0)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_message_strobe_0, 'strobe'), (self.epy_block_0, 'frame_trigger'))
        self.msg_connect((self.blocks_message_strobe_0, 'strobe'), (self.epy_block_0_0, 'frame_trigger'))
        self.msg_connect((self.epy_block_0, 'pdu_out'), (self.epy_block_0_1, 'pdus'))
        self.msg_connect((self.epy_block_0_0, 'pdu_out'), (self.epy_block_0_1_0, 'pdus'))
        self.connect((self.analog_const_source_x_0, 0), (self.blocks_null_sink_1, 0))
        self.connect((self.analog_const_source_x_0_0_0, 0), (self.blocks_float_to_complex_0_0_0, 1))
        self.connect((self.analog_const_source_x_0_0_0_0, 0), (self.blocks_float_to_complex_0_0_0_0, 1))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_throttle2_0_0, 0))
        self.connect((self.blocks_float_to_complex_0_0_0, 0), (self.blocks_multiply_xx_0_0_0, 1))
        self.connect((self.blocks_float_to_complex_0_0_0_0, 0), (self.blocks_multiply_xx_0_0_0_0, 1))
        self.connect((self.blocks_multiply_xx_0_0_0, 0), (self.fft_filter_xxx_0_0_0_0_0, 0))
        self.connect((self.blocks_multiply_xx_0_0_0_0, 0), (self.fft_filter_xxx_0_0_0_0_0_0, 0))
        self.connect((self.blocks_repeat_0_0_0, 0), (self.blocks_float_to_complex_0_0_0, 0))
        self.connect((self.blocks_repeat_0_0_0_0, 0), (self.blocks_float_to_complex_0_0_0_0, 0))
        self.connect((self.blocks_throttle2_0_0, 0), (self.root_raised_cosine_filter_0, 0))
        self.connect((self.blocks_uchar_to_float_0_1, 0), (self.blocks_repeat_0_0_0, 0))
        self.connect((self.blocks_uchar_to_float_0_1_0, 0), (self.blocks_repeat_0_0_0_0, 0))
        self.connect((self.digital_constellation_modulator_0_0_0_0, 0), (self.blocks_multiply_xx_0_0_0, 0))
        self.connect((self.digital_constellation_modulator_0_0_0_0_0, 0), (self.blocks_multiply_xx_0_0_0_0, 0))
        self.connect((self.epy_block_0_1, 1), (self.blocks_uchar_to_float_0_1, 0))
        self.connect((self.epy_block_0_1, 0), (self.digital_constellation_modulator_0_0_0_0, 0))
        self.connect((self.epy_block_0_1_0, 1), (self.blocks_uchar_to_float_0_1_0, 0))
        self.connect((self.epy_block_0_1_0, 0), (self.digital_constellation_modulator_0_0_0_0_0, 0))
        self.connect((self.fft_filter_xxx_0_0_0_0_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.fft_filter_xxx_0_0_0_0_0_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.fft_filter_xxx_0_0_0_0_0_0_0, 0), (self.blocks_file_sink_0_0, 0))
        self.connect((self.root_raised_cosine_filter_0, 0), (self.fft_filter_xxx_0_0_0_0_0_0_0, 0))
        self.connect((self.root_raised_cosine_filter_0, 0), (self.qtgui_const_sink_x_0_0, 0))
        self.connect((self.root_raised_cosine_filter_0, 0), (self.qtgui_freq_sink_x_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "TX_c")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_T_slot(((self.packet_size)*8*self.sps)/(self.bps*self.samp_rate))
        self.set_bw((1+self.excess_bw)*(self.samp_rate//self.sps))
        self.blocks_repeat_0_0_0.set_interpolation((8*self.sps))
        self.blocks_repeat_0_0_0_0.set_interpolation((8*self.sps))
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, ((self.samp_rate//self.sps)), 0.35, (11*self.sps)))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_T_slot(((self.packet_size)*8*self.sps)/(self.bps*self.samp_rate))
        self.set_bw((1+self.excess_bw)*(self.samp_rate//self.sps))
        self.set_low_pass_filter_taps(firdes.low_pass(1.0, self.samp_rate, (self.bw//2)+5e3, 10e3, window.WIN_HAMMING, 6.76))
        self.blocks_throttle2_0_0.set_sample_rate(self.samp_rate)
        self.qtgui_freq_sink_x_0_0.set_frequency_range(0, self.samp_rate)
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, ((self.samp_rate//self.sps)), 0.35, (11*self.sps)))

    def get_excess_bw(self):
        return self.excess_bw

    def set_excess_bw(self, excess_bw):
        self.excess_bw = excess_bw
        self.set_bw((1+self.excess_bw)*(self.samp_rate//self.sps))

    def get_packet_size(self):
        return self.packet_size

    def set_packet_size(self, packet_size):
        self.packet_size = packet_size
        self.set_T_slot(((self.packet_size)*8*self.sps)/(self.bps*self.samp_rate))
        self.epy_block_0.packet_size = self.packet_size
        self.epy_block_0_0.packet_size = self.packet_size

    def get_bw(self):
        return self.bw

    def set_bw(self, bw):
        self.bw = bw
        self.set_low_pass_filter_taps(firdes.low_pass(1.0, self.samp_rate, (self.bw//2)+5e3, 10e3, window.WIN_HAMMING, 6.76))

    def get_bps(self):
        return self.bps

    def set_bps(self, bps):
        self.bps = bps
        self.set_T_slot(((self.packet_size)*8*self.sps)/(self.bps*self.samp_rate))

    def get_usrp_rate(self):
        return self.usrp_rate

    def set_usrp_rate(self, usrp_rate):
        self.usrp_rate = usrp_rate

    def get_user_id2(self):
        return self.user_id2

    def set_user_id2(self, user_id2):
        self.user_id2 = user_id2
        self.epy_block_0_0.user_id = self.user_id2

    def get_user_id1(self):
        return self.user_id1

    def set_user_id1(self, user_id1):
        self.user_id1 = user_id1
        self.epy_block_0.user_id = self.user_id1

    def get_thresh(self):
        return self.thresh

    def set_thresh(self, thresh):
        self.thresh = thresh

    def get_taps(self):
        return self.taps

    def set_taps(self, taps):
        self.taps = taps

    def get_rs_ratio(self):
        return self.rs_ratio

    def set_rs_ratio(self, rs_ratio):
        self.rs_ratio = rs_ratio

    def get_phase_bw(self):
        return self.phase_bw

    def set_phase_bw(self, phase_bw):
        self.phase_bw = phase_bw

    def get_low_pass_filter_taps(self):
        return self.low_pass_filter_taps

    def set_low_pass_filter_taps(self, low_pass_filter_taps):
        self.low_pass_filter_taps = low_pass_filter_taps
        self.fft_filter_xxx_0_0_0_0_0.set_taps(self.low_pass_filter_taps)
        self.fft_filter_xxx_0_0_0_0_0_0.set_taps(self.low_pass_filter_taps)
        self.fft_filter_xxx_0_0_0_0_0_0_0.set_taps(self.low_pass_filter_taps)

    def get_initial_delay(self):
        return self.initial_delay

    def set_initial_delay(self, initial_delay):
        self.initial_delay = initial_delay
        self.epy_block_0.initial_delay = self.initial_delay
        self.epy_block_0_0.initial_delay = self.initial_delay

    def get_dist2(self):
        return self.dist2

    def set_dist2(self, dist2):
        self.dist2 = dist2
        self.epy_block_0_0.lambda_dist = self.dist2

    def get_dist1(self):
        return self.dist1

    def set_dist1(self, dist1):
        self.dist1 = dist1
        self.epy_block_0.lambda_dist = self.dist1

    def get_bpsk(self):
        return self.bpsk

    def set_bpsk(self, bpsk):
        self.bpsk = bpsk

    def get_T_slot(self):
        return self.T_slot

    def set_T_slot(self, T_slot):
        self.T_slot = T_slot
        self.blocks_message_strobe_0.set_period(((int)(self.N_Slots * self.T_slot * 1e3)))
        self.epy_block_0.slot_duration = self.T_slot
        self.epy_block_0_0.slot_duration = self.T_slot

    def get_N_Slots(self):
        return self.N_Slots

    def set_N_Slots(self, N_Slots):
        self.N_Slots = N_Slots
        self.blocks_message_strobe_0.set_period(((int)(self.N_Slots * self.T_slot * 1e3)))
        self.epy_block_0.N_slots = self.N_Slots
        self.epy_block_0_0.N_slots = self.N_Slots




def main(top_block_cls=TX_c, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

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
