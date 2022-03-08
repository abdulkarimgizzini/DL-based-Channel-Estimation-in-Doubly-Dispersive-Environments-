function [H_STA, Equalized_OFDM_Symbols] = DPA_TA(he_LS_Preamble ,y_r, Kset,mod, nUSC, nSym, ppositions)

%[subcarriers, OFDM_Symbols] = size(y_r(Kset,:));
% H_STA0 = he_LS_Preamble;
% Received_OFDM_Symbols = y_r(Kset,:);
H_STA = zeros(nUSC, nSym);
%H_STA_step4 = zeros(subcarriers, OFDM_Symbols);
Equalized_OFDM_Symbols = zeros(nUSC, nSym);
%ppositions = [7,21, 32,46].';

for i = 1:nSym
    if(i == 1)
        % Step 1: First, the ith received symbol, Received_OFDM_Symbols(i,k) is equalized by the channel estimate, 
        % of the previous symbol. For i = 1, STA(0) is the LS estimate at
        % the preamble.
        Equalized_OFDM_Symbol = y_r(Kset,i)./ he_LS_Preamble;
        Equalized_OFDM_Symbols(:,i) = Equalized_OFDM_Symbol;
        % Step 2.1: Equalized_OFDM_Symbol is demodulated to obtain
        % De_Equalized_OFDM_Symbol.
        % De_Equalized_OFDM_Symbol = qamdemod(sqrt(10) * Equalized_OFDM_Symbol ,16);
        De_Equalized_OFDM_Symbol = wlanClosestReferenceSymbol(Equalized_OFDM_Symbol,mod);
        De_Equalized_OFDM_Symbol(ppositions,:) = [1;1;1;-1]; 
        % Step 2.2: the ith received symbol is divided by
        % De_Equalized_OFDM_Symbol to obtain the initial channel estimate
        Initial_Channel_Estimate = y_r(Kset,i)./ De_Equalized_OFDM_Symbol;
        
       
        % Step 4: Time domain Averging
        STA_TA = (1 - (1/2)) .* he_LS_Preamble + (1/2).* Initial_Channel_Estimate;
        
        % Update H_STA Matrix
        H_STA(:,i) = STA_TA; 
        
    elseif (i > 1)
        % Step 1: First, the ith received symbol, Received_OFDM_Symbols(i,k) is equalized by the channel estimate, 
        % of the previous symbol. 
        Equalized_OFDM_Symbol = y_r(Kset,i)./ H_STA(:,i-1);
        Equalized_OFDM_Symbols(:,i) = Equalized_OFDM_Symbol;
        % Step 2.1: Equalized_OFDM_Symbol is demodulated to obtain
        % De_Equalized_OFDM_Symbol.
        De_Equalized_OFDM_Symbol = wlanClosestReferenceSymbol(Equalized_OFDM_Symbol,mod);
        De_Equalized_OFDM_Symbol(ppositions,:) = [1;1;1;-1]; 
        %De_Equalized_OFDM_Symbol(ppositions,:) = Received_OFDM_Symbols(ppositions,i) ./ [1;1;1;-1]; 
        
        % Step 2.2: the ith received symbol is divided by
        % De_Equalized_OFDM_Symbol to obtain the initial channel estimate
        Initial_Channel_Estimate = y_r(Kset,i)./ De_Equalized_OFDM_Symbol;
       
        % Step 4: Time domain Averging
        STA_TA = (1 - (1/2)) .*  H_STA(:,i-1) + (1/2).* Initial_Channel_Estimate;
        
        % Update H_STA Matrix
        H_STA(:,i) = STA_TA; 
    end
end

end

