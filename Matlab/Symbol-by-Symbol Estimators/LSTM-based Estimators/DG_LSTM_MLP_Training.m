clc;clearvars;close all; warning('off','all');
%% Simulation Case
algo = 'DPA_TA';
DL   = 'LSTM';
nSC_In                 = 104;% 104
snrr = 40;
mobility = 'H';
modu = 'QPSK';


load(['D:\ChPrediction\',mobility,'\',modu,'\Training_Simulation_',num2str(snrr),'.mat'],...
   'True_Channels_Structure', [algo '_Structure'],'HLS_Structure');

True_Channels_Structure = True_Channels_Structure(:,2:end,:);

ppositions             = [7,21, 32,46].';  
dpositions             = [1:6, 8:20, 22:31, 33:45, 47:52].';        
Training_Data_set_size = 8000;
nSym                   = 50;
nSC_Out                = 96;
Train_X                = zeros(nSC_In,nSym,Training_Data_set_size);
Train_Y                = zeros(nSC_Out,nSym,Training_Data_set_size);
Algo_Channels_Structure = eval([algo '_Structure']);


RPl = reshape(real(Algo_Channels_Structure(ppositions,1,:)),4*1,Training_Data_set_size);
IPl = reshape(imag(Algo_Channels_Structure(ppositions,1,:)),4*1,Training_Data_set_size);

RPL = real(Algo_Channels_Structure(ppositions,2:end,:));
IPL = imag(Algo_Channels_Structure(ppositions,2:end,:));
RHP = real(Algo_Channels_Structure(:,1:end-1,:));
IHP = imag(Algo_Channels_Structure(:,1:end-1,:));

if(nSC_In == 112)
    Train_X(:,1,:)     = [real(HLS_Structure); RPl; imag(HLS_Structure); IPl];
    Train_X(:,2:end,:) = [RHP; RPL; IHP; IPL];
elseif(nSC_In == 104)
    Train_X(:,1,:)     = [real(HLS_Structure); imag(HLS_Structure)];
    Train_X(:,2:end,:) = [RHP; IHP];
end

Train_Y(1:48,:,:)  = real(True_Channels_Structure(dpositions,:,:));
Train_Y(49:96,:,:) = imag(True_Channels_Structure(dpositions,:,:));

Train_X = permute(Train_X, [3, 2 ,1 ]);
Train_Y = permute(Train_Y, [3 2 1]);
Channel_Error_Correction_Dataset.('Train_X') =  Train_X;
Channel_Error_Correction_Dataset.('Train_Y') =  Train_Y;
save(['D:\ChPrediction\',mobility,'\',modu,'\',algo,'_', DL,'_Training_Dataset_',num2str(snrr)],  'Channel_Error_Correction_Dataset');

