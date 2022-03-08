clc;clearvars;close all; warning('off','all');
%% Simulation Case
algo = 'DPA_TA';
DL   = 'LSTM';
mobility = 'H';
modu = 'QPSK';
Testing_Data_set_size = 2000;
snrr = 0:5:40;
nSym = 50;
ppositions             = [7,21, 32,46].';  
dpositions             = [1:6, 8:20, 22:31, 33:45, 47:52].';

nSC_In                 = 104;% 104
nSC_Out                = 96;
for n_snr = 1:size(snrr,2)
    
Test_X = zeros(nSC_In, nSym, Testing_Data_set_size);
Test_Y = zeros(nSC_Out, nSym, Testing_Data_set_size);


load(['D:\ChPrediction\',mobility,'\',modu,'\Testing_Simulation_',num2str(snrr(n_snr)),'.mat'],...
   'True_Channels_Structure','Received_Symbols_FFT_Structure', [algo '_Structure'],'HLS_Structure');

TRFI_Channels_Structure = eval([algo '_Structure']);
True_Channels_Structure = True_Channels_Structure(:,2:end,1:Testing_Data_set_size);

Received_Symbols_FFT_Structure  =  Received_Symbols_FFT_Structure(dpositions,:,1:Testing_Data_set_size);

Testing_DatasetX =  TRFI_Channels_Structure;
Testing_DatasetY =  True_Channels_Structure;

HLS_Structure_Testing = HLS_Structure(:,1:Testing_Data_set_size);

RPl = reshape(real(Testing_DatasetX(ppositions,1,1:Testing_Data_set_size)),4*1,Testing_Data_set_size);
IPl = reshape(imag(Testing_DatasetX(ppositions,1,1:Testing_Data_set_size)),4*1,Testing_Data_set_size);


RPL = real(Testing_DatasetX(ppositions,2:end,1:Testing_Data_set_size));
IPL = imag(Testing_DatasetX(ppositions,2:end,1:Testing_Data_set_size));
RPP = real(Testing_DatasetX(ppositions,1:end-1,1:Testing_Data_set_size));
IPP = imag(Testing_DatasetX(ppositions,1:end-1,1:Testing_Data_set_size));

if(nSC_In == 112)
    Test_X(:,1,:) = [real(HLS_Structure_Testing); RPl; imag(HLS_Structure_Testing); IPl];
    
    Test_X(ppositions,2:end,:) = RPP;
    Test_X(53:56,2:end,:) = RPL;
    Test_X([63;77;88;102],2:end,:) = IPP;
    Test_X(109:112,2:end,:) = IPL;

elseif(nSC_In == 104)
    Test_X(:,1,:) = [real(HLS_Structure_Testing); imag(HLS_Structure_Testing)];
    Test_X(ppositions,2:end,:) = RPP;
    Test_X(ppositions + 52,2:end,:) = IPP;


end

Test_Y(1:48,:,:)  = real(Testing_DatasetY(dpositions,:,1:Testing_Data_set_size));
Test_Y(49:96,:,:) = imag(Testing_DatasetY(dpositions,:,1:Testing_Data_set_size));

Test_X = permute(Test_X,[3 2 1]);
Test_Y = permute(Test_Y,[3 2 1]);
Received_Symbols_FFT_Structure = permute(Received_Symbols_FFT_Structure,[3 2 1]);
Channel_Error_Correction_Dataset.('Test_X') =  Test_X;
Channel_Error_Correction_Dataset.('Test_Y') =  Test_Y;
Channel_Error_Correction_Dataset.('Y_DataSubCarriers') =  Received_Symbols_FFT_Structure;
save(['D:\ChPrediction\',mobility,'\',modu,'\',algo,'_',DL,'_Testing_Dataset_' num2str(n_snr)],  'Channel_Error_Correction_Dataset');

end