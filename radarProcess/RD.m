%% 手势识别RD图
close all;
clear;
clc;
%% c初始化
Frames = 32;  %帧数
PFA=1e-6;
sss = {'houyi','latui','qiantui','tuila','tuiyou','tuizuo','youhua','youzuohua','zuohua','zuoyouhua'};
for ss = sss
    %% 路径解析
    path = char(ss);
%     path = ['houyi'];
    list = dir(['D:\workplace\chongqing\GestureRecognition\DataSet\test2\matDataSet\',path ,'\','*.mat']);
    k = length(list);
    path2 = ['D:\workplace\chongqing\GestureRecognition\DataSet\test2\RdDataSet\',path ];
    mkdir(path2);

    %% 计算
    for i = 1:1:k
        str = strcat('D:\workplace\chongqing\GestureRecognition\DataSet\test2\matDataSet\',path,'\',list(i).name);
        load(str);
        DataSize = size(adcData);   %数据大小
        numsf = DataSize(2)/Frames; %每帧的采样点数
        numst = 256;                %快时间点数
        adcDatas = sum(adcData);

        path3 = [path2,'\',num2str(i-1, '%04d')];
        mkdir(path3);

        for n = 1:1:Frames  %对每一帧数据进行处理,共32帧
            adcDataFrame = adcDatas(1, numsf*(n-1)+1 : numsf*n);
            prtData = zeros(64, 256);
            for m = 1:1:64 %对每PRT数据进行处理,共64个prt
                Data = adcDataFrame(numst*(m-1)+1 : numst*m);
                prtData(m,:) = Data; %每个prt数据
            end

            % MTI滑动对消
            mtiData = zeros(63, 256);
            for m = 1:1:64-1
                mtiData(m, :) = prtData(m+1,:) - prtData(m,:);
            end

            %二维FFT画RD图
            fftData = fftshift(fft2(mtiData),1);
            %CFAR检测
            cfarData = zeros(63, 256);
            for m=1:64-1   
                for k=1:256
                    if k<8
                        a=[fftData(m,1:k-2),fftData(m,k+2:k+5)];  %滑窗估算噪声功率,窗口大小为11个点
                    elseif k>249
                        a=[fftData(m,k-5:k-2),fftData(m,k+2:256)];
                    else
                         a=[fftData(m,k-5:k-2),fftData(m,k+2:k+5)];
                    end
                    delta=mean(abs(a).^2);
                    t=sqrt(-delta*log(PFA));  %设置门限
                    if abs(fftData(m,k))>=t
                        cfarData(m,k)=fftData(m,k);  %将目标点存在新矩阵中
                    end
                end
            end
%             pause(0.00001);
            rd=abs(cfarData(:,1:63));
    %         finalData=abs(cfarData(11:46,1:16));
    %         maxData = max(max(finalData));
    %         minData = min(min(finalData));
    %         finalData = (finalData - minData)./(maxData - minData);
    %         finalData = imresize(finalData,[112,112]);
%             path4 = [path3,'\',num2str(n-1, '%02d'),'.png'];
            path4 = [path3,'\',num2str(n-1, '%02d'), '.mat'];
            save(path4,'rd');
%             imwrite((abs(finalData)),path4);
    %         imagesc((abs(rd)));
    %         hold on;
        end
        disp(['The ',num2str(i),' is finished!']);
    end
end


    


