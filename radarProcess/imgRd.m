%% ����ʶ��RDͼ
close all;
clear;
clc;
%% c��ʼ��
Frames = 32;  %֡��
PFA=1e-4;
% sss = {'houyi','latui','qiantui','tuila','tuiyou','tuizuo','youhua','youzuohua','zuohua','zuoyouhua'};
sss = {'test'};
for ss = sss
    %% ·������
    path = char(ss);
%     path = ['houyi'];
    list = dir(['D:\workplace\chongqing\GestureRecognition\DataSet\matDataSet\',path ,'\','*.mat']);
    k = length(list);
    path2 = ['D:\workplace\chongqing\GestureRecognition\DataSet\bothDataSet\',path ];
    mkdir(path2);

    %% ����
    for i = 1:1:k
        str = strcat('D:\workplace\chongqing\GestureRecognition\DataSet\matDataSet\',path,'\',list(i).name);
        load(str);
        DataSize = size(adcData);   %���ݴ�С
        numsf = DataSize(2)/Frames; %ÿ֡�Ĳ�������
        numst = 256;                %��ʱ�����
        adcDatas = sum(adcData);

        path3 = [path2,'\',num2str(i-1, '%04d')];
        mkdir(path3);

        for n = 1:1:Frames  %��ÿһ֡���ݽ��д���,��32֡
            adcDataFrame = adcDatas(1, numsf*(n-1)+1 : numsf*n);
            prtData = zeros(64, 256);
            for m = 1:1:64 %��ÿPRT���ݽ��д���,��64��prt
                Data = adcDataFrame(numst*(m-1)+1 : numst*m);
                prtData(m,:) = Data; %ÿ��prt����
            end

            % MTI��������
            mtiData = zeros(63, 256);
            for m = 1:1:64-1
                mtiData(m, :) = prtData(m+1,:) - prtData(m,:);
            end

            %��άFFT��RDͼ
            fftData = fftshift(fft2(mtiData),1);
            %CFAR���
            cfarData = zeros(63, 256);
            for m=1:64-1   
                for k=1:256
                    if k<8
                        a=[fftData(m,1:k-2),fftData(m,k+2:k+5)];  %����������������,���ڴ�СΪ11����
                    elseif k>249
                        a=[fftData(m,k-5:k-2),fftData(m,k+2:256)];
                    else
                         a=[fftData(m,k-5:k-2),fftData(m,k+2:k+5)];
                    end
                    delta=mean(abs(a).^2);
                    t=sqrt(-delta*log(PFA));  %��������
                    if abs(fftData(m,k))>=t
                        cfarData(m,k)=fftData(m,k);  %��Ŀ�������¾�����
                    end
                end
            end
%             pause(0.00001);
%             finalData=abs(cfarData);
            finalData=abs(cfarData(15:46,1:16));
%             finalData=abs(cfarData(11:46,1:16));
            maxData = max(max(finalData));
            minData = min(min(finalData));
            finalData = (finalData - minData)./(maxData - minData);
            finalData = imresize(finalData,[112,112]);
            path4 = [path3,'\',num2str(n-1, '%02d'),'.png'];
%             imwrite(abs(finalData),path4);   
            imagesc((abs(finalData)));
            set(gca,'XTick',[]) % Remove the ticks in the x axis!
            set(gca,'YTick',[]) % Remove the ticks in the y axis
            set(gca,'Position',[0 0 1 1]) % Make the axes occupy the hole figure
            saveas(gcf,path4)
            hold on;
        end
        disp(['The ',num2str(i),' is finished!']);
        close all;
    end
end


    


