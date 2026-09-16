#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: venkat
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
import pmt
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
import sip
import threading
import tx_1_1_epy_block_0 as epy_block_0  # embedded python block
import tx_1_1_epy_block_1 as epy_block_1  # embedded python block
import tx_1_1_epy_block_2 as epy_block_2  # embedded python block
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation




class tx_1_1(gr.top_block, Qt.QWidget):

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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "tx_1_1")

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
        self.samp_rate = samp_rate = 40000
        self.totalPkts = totalPkts = 2
        self.sps = sps = 10
        self.slots = slots = 3
        self.pktSize = pktSize = 32
        self.low_pass_filter_taps = low_pass_filter_taps = firdes.low_pass(1.0, samp_rate, 7.7e3, 10e3, window.WIN_HAMMING, 6.76)
        self.dist = dist = [(1.0, 2)]
        self.bpsk = bpsk = digital.constellation_bpsk().base()
        self.bpsk.set_npwr(1.0)

        ##################################################
        # Blocks
        ##################################################

        self.qtgui_time_sink_x_0 = qtgui.time_sink_c(
            1024, #size
            samp_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(True)
        self.qtgui_time_sink_x_0.enable_grid(True)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
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


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_0.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_0.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_win)
        self.fft_filter_xxx_0 = filter.fft_filter_ccc(1, low_pass_filter_taps, 1)
        self.fft_filter_xxx_0.declare_sample_delay(0)
        self.epy_block_2 = epy_block_2.blk()
        self.epy_block_1 = epy_block_1.blk(fs=samp_rate, sps=sps, pktSize=pktSize, slots=slots, silenceFrames=1)
        self.epy_block_0 = epy_block_0.blk(userId=1, slots=slots, TotalPkts=totalPkts, lambda_dist=dist, activeProb=1, pktSize=pktSize)
        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=bpsk,
            differential=True,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=0.35,
            verbose=False,
            log=False,
            truncate=False)
        self.blocks_throttle2_0 = blocks.throttle( gr.sizeof_gr_complex*1, samp_rate, True, 0 if "auto" == "auto" else max( int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1) )
        self.blocks_message_strobe_0 = blocks.message_strobe(pmt.intern("TEST"), 1000)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_message_strobe_0, 'strobe'), (self.epy_block_0, 'Trigger'))
        self.msg_connect((self.epy_block_0, 'PDU'), (self.epy_block_2, 'PDU_in'))
        self.connect((self.digital_constellation_modulator_0, 0), (self.epy_block_1, 0))
        self.connect((self.epy_block_1, 0), (self.fft_filter_xxx_0, 0))
        self.connect((self.epy_block_2, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.fft_filter_xxx_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.fft_filter_xxx_0, 0), (self.qtgui_time_sink_x_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "tx_1_1")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_low_pass_filter_taps(firdes.low_pass(1.0, self.samp_rate, 7.7e3, 10e3, window.WIN_HAMMING, 6.76))
        self.blocks_throttle2_0.set_sample_rate(self.samp_rate)
        self.epy_block_1.fs = self.samp_rate
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)

    def get_totalPkts(self):
        return self.totalPkts

    def set_totalPkts(self, totalPkts):
        self.totalPkts = totalPkts
        self.epy_block_0.TotalPkts = self.totalPkts

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.epy_block_1.sps = self.sps

    def get_slots(self):
        return self.slots

    def set_slots(self, slots):
        self.slots = slots
        self.epy_block_0.slots = self.slots
        self.epy_block_1.slots = self.slots

    def get_pktSize(self):
        return self.pktSize

    def set_pktSize(self, pktSize):
        self.pktSize = pktSize
        self.epy_block_0.pktSize = self.pktSize
        self.epy_block_1.pktSize = self.pktSize

    def get_low_pass_filter_taps(self):
        return self.low_pass_filter_taps

    def set_low_pass_filter_taps(self, low_pass_filter_taps):
        self.low_pass_filter_taps = low_pass_filter_taps
        self.fft_filter_xxx_0.set_taps(self.low_pass_filter_taps)

    def get_dist(self):
        return self.dist

    def set_dist(self, dist):
        self.dist = dist

    def get_bpsk(self):
        return self.bpsk

    def set_bpsk(self, bpsk):
        self.bpsk = bpsk




def main(top_block_cls=tx_1_1, options=None):

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
