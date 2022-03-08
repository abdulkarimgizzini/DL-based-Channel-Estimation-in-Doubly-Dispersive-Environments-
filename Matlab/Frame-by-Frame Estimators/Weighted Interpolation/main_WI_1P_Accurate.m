clc;clearvars;close all; warning('off','all');
ch_func = Channel_functions();
est_func = Estimation_functions();
%% --------OFDM Parameters - Given in IEEE 802.11p Spec--
ofdmBW                 = 10 * 10^6 ;     % OFDM bandwidth (Hz)
nFFT                   = 64;             % FFT size 
nDSC                   = 52;             % Number of data subcarriers
nPSC                   = 0;              % Number of pilot subcarriers
nZSC                   = 12;             % Number of zeros subcarriers
nUSC                   = nDSC + nPSC;    % Number of total used subcarriers
K                      = nUSC + nZSC;    % Number of total subcarriers
nSym                   = 100;             % Number of OFDM symbols within one frame
deltaF                 = ofdmBW/nFFT;    % Bandwidth for each subcarrier - include all used and unused subcarriers 
Tfft                   = 1/deltaF;       % IFFT or FFT period = 6.4us
Tgi                    = Tfft/4;         % Guard interval duration - duration of cyclic prefix - 1/4th portion of OFDM symbols = 1.6us
Tsignal                = Tgi+Tfft;       % Total duration of BPSK-OFDM symbol = Guard time + FFT period = 8us
K_cp                   = nFFT*Tgi/Tfft;  % Number of symbols allocated to cyclic prefix 
% Pre-defined preamble in frequency domain
dp = [ 0  0 0 0 0 0 +1 +1 -1 -1 +1  +1 -1  +1 -1 +1 +1 +1 +1 +1 +1 -1 -1 +1 +1 -1 +1 -1 +1 +1 +1 +1 0 +1 -1 -1 +1 +1 -1 +1 -1 +1 -1 -1 -1 -1 -1 +1 +1 -1 -1 +1 -1 +1 -1 +1 +1 +1 +1 0 0 0 0 0];
Ep                     = 1;              % pramble power per sample
dp                     = fftshift(dp);   % Shift zero-frequency component to center of spectrum    
predefined_preamble    = dp;
Kset                   = find(dp~=0);               % set of allocated subcarriers                  
Kon                    = length(Kset);              % Number of active subcarriers
dp                     = sqrt(Ep)*dp.';
xp                     = sqrt(K)*ifft(dp);
xp_cp                  = [xp(end-K_cp+1:end); xp];  
preamble_80211p        = repmat(xp_cp,1,2);  
p1                     = xp_cp;         
used_locations         = [2:27, 39:64].';
pilots_locations       = [8,22,44,58].'; % Pilot subcarriers positions
pilots                 = [1 1 1 -1].';
data_locations         = [2:7, 9:21, 23:27, 39:43, 45:57, 59:64].'; % Data subcarriers positions
ppositions             = [7,21, 32,46].';                           % Pilots positions in Kset
dpositions             = [1:6, 8:20, 22:31, 33:45, 47:52].';        % Data positions in Kset
pilots_locations_Full  = [2,6,12,18,24,27,39,45,51,57,61,64].'; % Pilot subcarriers positions
%% ------ Bits Modulation Technique------------------------------------------
mod                       = 'QPSK';
Mod_Type                  = 1;              % 0 for BPSK and 1 for QAM 
if(Mod_Type == 0)
    nBitPerSym            = 1;
    Pow                   = 1;
    %BPSK Modulation Objects
    bpskModulator         = comm.BPSKModulator;
    bpskDemodulator       = comm.BPSKDemodulator;
    M                     = 1;
elseif(Mod_Type == 1)
    if(strcmp(mod,'QPSK') == 1)
         nBitPerSym            = 2; 
    elseif (strcmp(mod,'16QAM') == 1)
         nBitPerSym            = 4; 
    elseif (strcmp(mod,'64QAM') == 1)
         nBitPerSym            = 6; 
    end
    %nBitPerSym            = 2; % Number of bits per Data Subcarrier (2 --> 4QAM, 4 --> 16QAM, 6 --> 64QAM)
    M                     = 2 ^ nBitPerSym; % QAM Modulation Order   
    Pow                   = mean(abs(qammod(0:(M-1),M)).^2); % Normalization factor for QAM        
end
%% ---------Scrambler Parameters---------------------------------------------
scramInit                 = 93; % As specidied in IEEE 802.11p Standard [1011101] in binary representation
%% ---------Convolutional Coder Parameters-----------------------------------
constlen                  = 7;
trellis                   = poly2trellis(constlen,[171 133]);
tbl                       = 34;
rate                      = 1/2;
%% -------Interleaver Parameters---------------------------------------------
% Matrix Interleaver
Interleaver_Rows          = 16;
Interleaver_Columns       = (nBitPerSym * nDSC * nSym) / Interleaver_Rows;
% General Block Interleaver
Random_permutation_Vector = randperm(nBitPerSym*nDSC*nSym); % Permutation vector
%% -----------------Vehicular Channel Model Parameters--------------------------
ChType                    = 'VTV_UC';               % Channel model
ch_l                      = [6 7 8 9];              % Select only Np Dominant multi path components for VTV_UC channel model
fs                        = K*deltaF;               % Sampling frequency in Hz, here case of 802.11p with 64 subcarriers and 156250 Hz subcarrier spacing
fc                        = 5.9e9;                  % Carrier Frequecy in Hz.
v                         = 45;                      % Moving speed of user in km/h
c                         = 3e8;                    % Speed of Light in m/s
fD                        = (v/3.6)/c*fc;           % Doppler freq in Hz
plotFlag                  = 0;                      % 1 to display the channel frequency response
rchan                     = ch_func.GenFadingChannel(ChType, fD, fs);
release(rchan);
init_seed = 22;
%% ---------Bit to Noise Ratio------------------%
EbN0dB                    = 0:5:30;         % bit to noise ratio
SNR_p                     = EbN0dB + 10*log10(K/nDSC) + 10*log10(K/(K + K_cp)) + 10*log10(nBitPerSym) + 10*log10(rate); % converting to symbol to noise ratio
SNR_p                     = SNR_p.';
N0                        = Ep*10.^(-SNR_p/10);

%% 1D DFT Interpolation
D                       = dftmtx (nFFT);
Dt_LS                   = D(Kset,ch_l+1);
DpLS                    = D(Kset,ch_l+1);
temp_LS = ((DpLS' * DpLS)^-1) * DpLS';
H_Interpolation_LS   = Dt_LS * temp_LS; 

% Full DFT
Dt_Full                   = D(Kset,ch_l+1);
Dp_Full                   = D(pilots_locations_Full,ch_l+1);
temp_Full                 = ((Dp_Full' * Dp_Full)^-1) * Dp_Full';
H_Interpolation_Full      = Dt_Full * temp_Full;
ERR_DFT                   = trace (H_Interpolation_Full * H_Interpolation_Full');

%% Simulation Parameters 
load('C:\Users\Abdulkarim Gizzini\Desktop\IEEE802.11p_TX_RX\IDX20000.mat');
training_indices = Training_IDX.';
Testing_Packets_Indices = Testing_IDX.';
Training_Data_set_size = size(training_indices,1);
Testing_Data_set_size = size(Testing_Packets_Indices,1);
N_CH                    = Training_Data_set_size; % 1000
N_SNR                   = length(SNR_p);

% Normalized mean square error (NMSE) vectors
Err_FP_ALS   = zeros(N_SNR,1);
Err_FP_SLS   = zeros(N_SNR,1);
Err_LP       = zeros(N_SNR,1);

% Bit error rate (BER) vectors
Ber_Ideal    = zeros(N_SNR,1);
Ber_FP_ALS   = zeros(N_SNR,1);
Ber_FP_SLS   = zeros(N_SNR,1);
Ber_LP       = zeros(N_SNR,1);
% average channel power E(|hf|^2)
Phf_H_Total             = zeros(N_SNR,1);
%% Simulation Loop
for n_snr = 1:N_SNR
    disp(['Running Simulation, SNR = ', num2str(EbN0dB(n_snr))]);
    tic;  
     rchan.Seed = 1234;
%       TX_Bits_Stream_Structure                 = zeros(nDSC * nSym  * nBitPerSym *rate, N_CH);
%       Received_Symbols_FFT_Structure           = zeros(Kon,nSym, N_CH);
%       True_Channels_Structure                  = zeros(Kon, nSym, N_CH);
%       WI_FP_SLS_Structure                      = zeros(Kon, nSym, N_CH);
%      WI_LP_Structure                          = zeros(Kon, nSym, N_CH);
%      WI_CP_Structure                          = zeros(Kon, nSym, N_CH);
    for n_ch = 1:N_CH 
        % Bits Stream Generation 
        Bits_Stream_Coded = randi(2, nDSC * nSym  * nBitPerSym * rate,1)-1;
        % Data Scrambler 
        scrambledData = wlanScramble(Bits_Stream_Coded,scramInit);
        % Convolutional Encoder
        dataEnc = convenc(scrambledData,trellis);
        % Interleaving
        % Matrix Interleaving
        codedata = dataEnc.';
        Matrix_Interleaved_Data = matintrlv(codedata,Interleaver_Rows,Interleaver_Columns).';
        % General Block Interleaving
        General_Block_Interleaved_Data = intrlv(Matrix_Interleaved_Data,Random_permutation_Vector);
        % Bits Mapping: M-QAM Modulation
        TxBits_Coded = reshape(General_Block_Interleaved_Data,nDSC , nSym  , nBitPerSym);
        % Gray coding goes here
        TxData_Coded = zeros(nDSC ,nSym);
        for m = 1 : nBitPerSym
           TxData_Coded = TxData_Coded + TxBits_Coded(:,:,m)*2^(m-1);
        end
        % M-QAM Modulation
         Modulated_Bits_Coded  =1/sqrt(Pow) * qammod(TxData_Coded,M);
        
         % OFDM Frame Generation
         OFDM_Frame_Coded = zeros(K,nSym);
         OFDM_Frame_Coded(used_locations,:) = Modulated_Bits_Coded;
         % Taking FFT, the term (nFFT/sqrt(nDSC)) is for normalizing the power of transmit symbol to 1 
         IFFT_Data_Coded =  sqrt(K)*ifft(OFDM_Frame_Coded); 
         % Appending cylic prefix
         CP_Coded = IFFT_Data_Coded((K - K_cp +1):K,:);
         IFFT_Data_CP_Coded = [CP_Coded; IFFT_Data_Coded];
         % Appending preamble symbol 
         IFFT_Data_CP_Preamble_Coded = [preamble_80211p IFFT_Data_CP_Coded p1];
         
 
        % ideal estimation
        [ h, y ] = ch_func.ApplyChannel( rchan, IFFT_Data_CP_Preamble_Coded, K_cp);
        release(rchan);
        rchan.Seed = rchan.Seed+1;
        
        yp = y((K_cp+1):end,1:2);
        yp1 = y((K_cp+1):end,end);
        y  = y((K_cp+1):end,3:end-1);
        
        
        yFD =  sqrt(1/K)*fft(y);
        yfp =  sqrt(1/K)*fft(yp); 
        yfp1 =  sqrt(1/K)*fft(yp1); 
      
        
        h = h((K_cp+1):end,:);
        hf = fft(h);    
        hf  = hf(:,3:end-1);

        
        Phf_H_Total(n_snr) = Phf_H_Total(n_snr) + mean(sum(abs(hf(Kset,:)).^2));
        %add noise
        noise_p = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,2], 1);
        noise_p1 = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,1], 1);
        yfp_r = yfp +  noise_p;
        yfp1_r = yfp1 +  noise_p1;
        noise_OFDM_Symbols = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,size(yFD,2)], 1);
        y_r   = yFD + noise_OFDM_Symbols;
       
         
       %% Channel Estimation
       % LS
       he_LS_P1 = ((yfp_r(Kset,1) + yfp_r(Kset,2))./(2.*predefined_preamble(Kset).'));
       he_LS_P2 = (yfp1_r(Kset,1)./ predefined_preamble(Kset).');
       % P Matrix Calculation
       P = [he_LS_P1 he_LS_P2].';

       % LS
       %LS_P1 =  ((yfp_r(Kset,1) + yfp_r(Kset,2))./(2.*predefined_preamble(Kset).'));
       %LS_P2 = (yfp1_r(Kset,1)./ predefined_preamble(Kset).');
       %P_LS = [LS_P1 LS_P2].';
       
       % DFT Interpolation
       %HLS_P2_Full = yfp1_r(pilots_locations_Full,1)./ predefined_preamble(pilots_locations_Full).';
       %H_DFT_Interpolation_ZF_P2_Full = H_Interpolation_Full * HLS_P2_Full; 
       %P1_DFT_Full = [he_LS_P1 H_DFT_Interpolation_ZF_P2_Full(:,1)].';

        
        
        % Interpolation Matrix Calculation
        H_Interpolated_Data_Frame = zeros(nDSC,nSym);
        %H_Interpolated_Data_Frame_MDFT = zeros(nDSC,nSym);
        %H_Interpolated_Data_Frame_DFT  = zeros(nDSC,nSym);
        noise_power_OFDM_Symbols = var(noise_OFDM_Symbols);
        [Ck_Cofficients] = Interpolation_Cofficients1(nSym, fD, Tsignal, noise_power_OFDM_Symbols);   
        %[Ck_Cofficients_DFT] = Interpolation_Cofficients_DFT(nSym, fD, Tsignal, noise_power_OFDM_Symbols, ERR_DFT);
      
        for li = 1:nSym 
            H_Interpolated_Data_Frame(:,li) = Ck_Cofficients(li,:) * P; 
        end
       err_LS_Interpolation = mean(sum(abs(H_Interpolated_Data_Frame - hf(Kset,:)).^2));
       Err_LS_Interpolation (n_snr) = Err_LS_Interpolation (n_snr) + err_LS_Interpolation;
       
      
    
       
       
      
        
          
       % Equalization
       y_Ideal = y_r(Kset ,:) ./ hf(Kset,:);
       y_LS_Interpolation = y_r(Kset ,:)./ H_Interpolated_Data_Frame;
       %y_LS_Interpolation_MDFT = y_r(Kset ,:)./ H_Interpolated_Data_Frame_MDFT;
       %y_LS_Interpolation_DFT = y_r(Kset ,:)./ H_Interpolated_Data_Frame_DFT;


       % QAM - DeMapping
       De_Mapped_Ideal = qamdemod(sqrt(Pow) * y_Ideal,M);
       De_Mapped_LS_Interpolation  = qamdemod(sqrt(Pow) * y_LS_Interpolation ,M);
       %De_Mapped_LS_Interpolation_MDFT  = qamdemod(sqrt(Pow) * y_LS_Interpolation_MDFT ,M);
       %De_Mapped_LS_Interpolation_DFT  = qamdemod(sqrt(Pow) * y_LS_Interpolation_DFT ,M);

       % Bits Extraction
       Bits_Ideal = zeros(nDSC,nSym,log2(M));
       Bits_LS_Interpolation  = zeros(nDSC,nSym,log2(M));
       %Bits_LS_Interpolation_MDFT  = zeros(nDSC,nSym,log2(M));
       %Bits_LS_Interpolation_DFT  = zeros(nDSC,nSym,log2(M));

          for b = 1 : nSym
                   Bits_Ideal(:,b,:) = de2bi(De_Mapped_Ideal(:,b));
                   Bits_LS_Interpolation(:,b,:) = de2bi(De_Mapped_LS_Interpolation(:,b)); 
        %           Bits_LS_Interpolation_MDFT(:,b,:) = de2bi(De_Mapped_LS_Interpolation_MDFT(:,b));
        %           Bits_LS_Interpolation_DFT(:,b,:) = de2bi(De_Mapped_LS_Interpolation_DFT(:,b));
          end           
       
       % De-Interleaving
       % General Block De-Interleaving
       General_Block_De_Interleaved_Data_Ideal = deintrlv(Bits_Ideal(:),Random_permutation_Vector);
       General_Block_De_Interleaved_Data_LS_Interpolation     = deintrlv(Bits_LS_Interpolation(:),Random_permutation_Vector);
       %General_Block_De_Interleaved_Data_LS_Interpolation_MDFT     = deintrlv(Bits_LS_Interpolation_MDFT(:),Random_permutation_Vector);
       %General_Block_De_Interleaved_Data_LS_Interpolation_DFT     = deintrlv(Bits_LS_Interpolation_DFT(:),Random_permutation_Vector);
       
       % Matrix De-Interleaving
       Matrix_De_Interleaved_Data_Ideal = matintrlv(General_Block_De_Interleaved_Data_Ideal.',Interleaver_Columns,Interleaver_Rows).';
       Matrix_De_Interleaved_Data_LS_Interpolation     = matintrlv(General_Block_De_Interleaved_Data_LS_Interpolation .',Interleaver_Columns,Interleaver_Rows).';
       %Matrix_De_Interleaved_Data_LS_Interpolation_MDFT     = matintrlv(General_Block_De_Interleaved_Data_LS_Interpolation_MDFT .',Interleaver_Columns,Interleaver_Rows).';
       %Matrix_De_Interleaved_Data_LS_Interpolation_DFT     = matintrlv(General_Block_De_Interleaved_Data_LS_Interpolation_DFT .',Interleaver_Columns,Interleaver_Rows).';   
       % Viterbi decoder
       Decoded_Bits_Ideal = vitdec(Matrix_De_Interleaved_Data_Ideal,trellis,tbl,'trunc','hard');
       Decoded_Bits_LS_Interpolation     = vitdec(Matrix_De_Interleaved_Data_LS_Interpolation ,trellis,tbl,'trunc','hard');
       %Decoded_Bits_LS_Interpolation_MDFT     = vitdec(Matrix_De_Interleaved_Data_LS_Interpolation_MDFT ,trellis,tbl,'trunc','hard');
       %Decoded_Bits_LS_Interpolation_DFT     = vitdec(Matrix_De_Interleaved_Data_LS_Interpolation_DFT ,trellis,tbl,'trunc','hard'); 
           
       % De-scrambler Data
       Bits_Ideal_Final                  = wlanScramble(Decoded_Bits_Ideal,scramInit);
       Bits_LS_Final_Interpolation       = wlanScramble(Decoded_Bits_LS_Interpolation ,scramInit);
       %Bits_LS_Final_Interpolation_MDFT  = wlanScramble(Decoded_Bits_LS_Interpolation_MDFT ,scramInit);
       %Bits_LS_Final_Interpolation_DFT  = wlanScramble(Decoded_Bits_LS_Interpolation_DFT ,scramInit);

       % BER and FER Calculation
       ber_Ideal                       = biterr(Bits_Ideal_Final,Bits_Stream_Coded);           
       ber_LS_Interpolation            = biterr(Bits_LS_Final_Interpolation,Bits_Stream_Coded);
       %ber_LS_Interpolation_MDFT       = biterr(Bits_LS_Final_Interpolation_MDFT,Bits_Stream_Coded);
       %ber_LS_Interpolation_DFT       = biterr(Bits_LS_Final_Interpolation_DFT,Bits_Stream_Coded);
       Ber_Ideal (n_snr)               = Ber_Ideal (n_snr) + ber_Ideal;
       Ber_LS_Interpolation(n_snr)     = Ber_LS_Interpolation(n_snr) + ber_LS_Interpolation ;  
       %Ber_MDFT_Interpolation (n_snr)  = Ber_MDFT_Interpolation (n_snr) + ber_LS_Interpolation_MDFT;
       %Ber_DFT_Interpolation (n_snr)   = Ber_DFT_Interpolation (n_snr) + ber_LS_Interpolation_DFT;
       
       
     
%           TX_Bits_Stream_Structure(:, n_ch) = Bits_Stream_Coded;
%           Received_Symbols_FFT_Structure(:,:,n_ch) = y_r(Kset ,:);
%           True_Channels_Structure(:,:,n_ch) = hf(Kset,:);
%           WI_FP_SLS_Structure(:,:,n_ch)  =  H_Interpolated_Data_Frame; 
%          WI_LP_Structure(:,:,n_ch)  =  H_Interpolated_Data_Frame_DFT; 
%          WI_CP_Structure(:,:,n_ch)  =  H_Interpolated_Data_Frame_MDFT; 
    end  
%           save(['D:\lm\WI_1P_16QAM_Simulation_' num2str(n_snr)],...
%           'TX_Bits_Stream_Structure',...
%           'Received_Symbols_FFT_Structure',...
%           'True_Channels_Structure',...
%           'WI_FP_SLS_Structure');
    toc;
end 
%% Bit Error Rate (BER)
BER_Ideal             = Ber_Ideal /(N_CH * nSym * nDSC * nBitPerSym);
BER_LS_Interpolation  = Ber_LS_Interpolation  / (N_CH * nSym * nDSC * nBitPerSym);
BER_MDFT_Interpolation = Ber_MDFT_Interpolation  / (N_CH * nSym * nDSC * nBitPerSym);
BER_DFT_Interpolation  = Ber_DFT_Interpolation  / (N_CH * nSym * nDSC * nBitPerSym);
%% Normalized Mean Square Error
Phf_H                 = Phf_H_Total/(N_CH);
ERR_LS_Interpolation  = Err_LS_Interpolation / (Phf_H * N_CH); 
ERR_MDFT_Interpolation  = Err_MDFT_Interpolation / (Phf_H * N_CH); 
ERR_DFT_Interpolation  = Err_DFT_Interpolation / (Phf_H * N_CH); 
%% Plotting Results
figure,
p1 = semilogy(EbN0dB, BER_Ideal,'k-o','LineWidth',2);
hold on;
p2 = semilogy(EbN0dB, BER_LS_Interpolation,'r--o','LineWidth',2);
hold on;
p3 = semilogy(EbN0dB, BER_MDFT_Interpolation,'b--o','LineWidth',2);
hold on;
p4 = semilogy(EbN0dB, BER_DFT_Interpolation,'g--o','LineWidth',2);
hold on;
grid on;
legend([p1(1),p2(1),p3(1),p4(1)],{'Perfect Channel','WI-Accurate LS','WI-Simple LS','WI-LP'});
xlabel('SNR(dB)');
ylim([8.36538461538462e-06 10^0])
ylabel('BER');
title([ ChType, ' Channel Model, V = ', num2str(v), ' Kmph; Modulation: ', mod]);
% save(['D:\lm\WI_1P_16QAM_Simulation_variables'],...
%         'predefined_preamble','mod','Kset','Random_permutation_Vector','fD','ChType');
%  

return
figure,
p1 = semilogy(EbN0dB, ERR_LS_Interpolation,'r--o','LineWidth',2);
hold on;
p2 = semilogy(EbN0dB, ERR_MDFT_Interpolation,'b--o','LineWidth',2);
hold on;
p3 = semilogy(EbN0dB, ERR_DFT_Interpolation,'g--o','LineWidth',2);
hold on;
grid on;
legend([p1(1),p2(1),p3(1)],{'WI-Accurate LS','WI-Simple LS','WI-LP'});
xlabel('SNR(dB)');
ylabel('NMSE');
title([ ChType, ' Channel Model, V = ', num2str(v), ' Kmph; Modulation: ', mod]);



 