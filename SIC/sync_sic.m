clc
clear
close all

filename = 'sync_samples.bin';

fileID = fopen(filename, 'rb');

if fileID == -1
    error("Can't open file");
end

data = fread(fileID, 'float32');
fclose(fileID);

if mod(length(data), 2) ~= 0
    data = data(1:end-1);
end

I = data(1:2:end);
Q = data(2:2:end);

iq_signal = I + 1j * Q;

fs = 40*1e3;
sps = 10;
alpha = 0.35;
span = 11;

rrc_taps = rcosdesign(alpha, span, sps, 'sqrt');

access_code_bits = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1];

