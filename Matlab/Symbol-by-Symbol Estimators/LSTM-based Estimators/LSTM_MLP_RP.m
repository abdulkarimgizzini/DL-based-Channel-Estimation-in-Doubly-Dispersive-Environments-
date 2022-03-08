clc;clearvars;close all; warning('off','all');
mobility  = 'H';
modu = 'QPSK';
load(['D:\ChPrediction\',mobility,'\',modu,'\Testing_Simulation_variables.mat']);
scheme                    = 'DPA';
DL                        = 'LSTM_DNN';
nBitPerSym                = 2;
M                         = 2 ^ nBitPerSym;
Pow                       = mean(abs(qammod(0:(M-1),M)).^2); 
Testing_Data_set_size     = 2000;
SNR_p                     = (0:5:40)';
EbN0dB                    = SNR_p;
nSym                      = 50;
Interleaver_Columns       = (nBitPerSym * 48 * nSym) / 16;
N_SNR                      = size(SNR_p,1);
Phf                        = zeros(N_SNR,1);
Err_AE_DNN  = zeros(N_SNR,1);
Ber_AE_DNN  = zeros(N_SNR,1);
dpositions             = [1:6, 8:20, 22:31, 33:45, 47:52].'; 
for i = 1:size(SNR_p,1) 
      disp(['Running Simulation, SNR = ', num2str(EbN0dB(i))]);
     tic;
     load(['D:\ChPrediction\',mobility,'\',modu,'\Testing_Simulation_',num2str(EbN0dB(i)),'.mat']);
     load(['D:\ChPrediction\',mobility,'\',modu,'\', scheme,'_',DL,'_Results_' num2str(i),'.mat']);

     TestY = eval([scheme,'_test_y_',num2str(i)]);
     TestY = permute(TestY,[3 2 1]);
     TestY = TestY(1:48,:,:) + 1i*TestY(49:96,:,:);
     
     PredictionY = eval([scheme,'_corrected_y_',num2str(i)]);
     PredictionY = permute(PredictionY,[3 2 1]);

  
   

    for u = 1:Testing_Data_set_size
          t = True_Channels_Structure(dpositions,2:end,u);
          t1 = TestY(:,:,u);

         H_AE_DNN = PredictionY(:,:,u);
 
        Phf(i) = Phf(i) + mean(sum(abs(True_Channels_Structure(dpositions,2:end,u)).^2)); 
        Err_AE_DNN (i) =  Err_AE_DNN (i) +  mean(sum(abs(H_AE_DNN - True_Channels_Structure(dpositions,2:end,u)).^2)); 
        
        % IEEE 802.11p Rx
        Bits_AE_DNN     = de2bi((qamdemod(sqrt(Pow) * (Received_Symbols_FFT_Structure(dpositions ,:,u) ./ H_AE_DNN),M)));
        %Bits_AE_DNN     = de2bi((qamdemod(sqrt(Pow) * (EqualizedS(:,:,u) ),M)));
        Ber_AE_DNN(i)   = Ber_AE_DNN(i) + biterr(wlanScramble(vitdec((matintrlv((deintrlv(Bits_AE_DNN(:),Random_permutation_Vector)).',Interleaver_Columns,16).'),poly2trellis(7,[171 133]),34,'trunc','hard'),93),TX_Bits_Stream_Structure(:,u));
    end
   toc;
end
Phf = Phf ./ Testing_Data_set_size;
ERR_AE_DNN = Err_AE_DNN ./ (Testing_Data_set_size * Phf); 
BER_AE_DNN = Ber_AE_DNN/ (Testing_Data_set_size * nSym * 48 * nBitPerSym);