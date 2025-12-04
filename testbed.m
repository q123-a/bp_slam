clear variables; clc;
close all;

% general parameters
parameters.known_track = 0;
%load('scenarioCleanM2_new901.mat');
 load('scenarioCleanM2_new901901.mat');
% cast visibilities to 1
[numSensors, ~] = size(dataVA);
for sensor = 1:numSensors
  dataVA{sensor}.visibility = ones(size(dataVA{sensor}.visibility,1),length(trueTrajectory));
end

% algoirthm parameters
parameters.maxSteps = 900;
trueTrajectory = trueTrajectory(:,1:parameters.maxSteps);
parameters.lengthStep = 0.03;
parameters.scanTime = 1;
v_max = parameters.lengthStep/parameters.scanTime;
parameters.drivingNoiseVariance = (v_max/3/parameters.scanTime)^2*0.5;
parameters.measurementVariance = .1^2;
parameters.measurementVarianceLHF = .15^2;
parameters.detectionProbability = 0.95;
parameters.regionOfInterestSize = 30;
parameters.meanNumberOfClutter = 1;
parameters.clutterIntensity = parameters.meanNumberOfClutter/parameters.regionOfInterestSize;
parameters.meanNumberOfBirth = 10^(-4);
parameters.birthIntensity = parameters.meanNumberOfBirth/(2*parameters.regionOfInterestSize)^2;
parameters.meanNumberOfUndetectedAnchors = 6;
parameters.undetectedAnchorsIntensity = parameters.meanNumberOfUndetectedAnchors/(2*parameters.regionOfInterestSize)^2;
parameters.numParticles = 100000;
parameters.upSamplingFactor = 1;

% SLAM parameters
parameters.detectionThreshold = 0.5;
parameters.survivalProbability = 0.999;
parameters.unreliabilityThreshold = 1e-4;
parameters.priorKnownAnchors{1} = 1;
parameters.priorKnownAnchors{2} = 1;
parameters.priorCovarianceAnchor = .001^2*eye(2);
parameters.anchorRegularNoiseVariance = 1e-4^2;

% agent parameters
parameters.UniformRadius_pos = .5;
parameters.UniformRadius_vel = .05;
% parameters.UniformRadius_pos = parameters.regionOfInterestSize; 
% parameters.UniformRadius_vel = 2;
% random seed
rng(1)

% draw starting position
parameters.priorMean = [trueTrajectory(1:2,1);0;0];
% parameters.priorMean = [0;0;0;0];
% generate measurements
%measurements = generateMeasurements(trueTrajectory, dataVA, parameters);

% generate cluttered measurements
%clutteredMeasurements = generateClutteredMeasurements(measurements, parameters);
%加载新数据


% --- 在您的主跟踪脚本中 ---

% 1. 定义物理常数和加载参数
% (您的跟踪脚本应该已经有了 'parameters' 结构体)
c = 3.0e8; % 光速 (m/s)
% 确保 'parameters.measurementVariance' 已经被定义
% 例如: parameters.measurementVariance = 0.5; (来自 generateMeasurements)

% 2. 加载 MVALSE 的输出 (这替换了 generateMeasurements 和 generateClutteredMeasurements)
disp('正在加载 MVALSE 的估计结果 (measurement256.mat)...');
%load('measurement512-3.mat'); % 加载 estimated_measurements_cell
% load('measurementbadf.mat'); 
load('measurementbadf.mat'); 
% 3. 初始化新的 clutteredMeasurements 变量
[numSteps, numSensors] = size(estimated_measurements_cell);
clutteredMeasurements = cell(numSteps, numSensors);

% % 4. 转换数据格式: (时延 -> 距离) 和 (设置方差)
% disp('正在将 MVALSE 的时延(s)输出转换为跟踪器期望的距离(m)...');
% 
% for step = 1:numSteps
%     for sensor = 1:numSensors
% 
%         % 从加载的文件中获取 MVALSE 的估计
%         mvalse_output = estimated_measurements_cell{step, sensor};
% 
%         % 检查 MVALSE 是否在该 (step, sensor) 上找到了任何路径
%         if ~isempty(mvalse_output)
% 
%             % 获取 MVALSE 找到的路径数
%             K_estimated = size(mvalse_output, 2);
% 
%             % 准备一个符合跟踪器格式的新矩阵
%             tracker_input = zeros(2, K_estimated);
% 
%             % --- 关键转换 (行 1) ---
%             % 转换: (行 1) 时延 [s] * 光速 [m/s] = 距离 [m]
%             tracker_input(1, :) = mvalse_output(1, :) * c;
% 
%             % --- 关键替换 (行 2) ---
%             % 替换: (行 2) 使用跟踪器期望的固定方差
%             % (注意: 我们丢弃了 mvalse_output(2,:), 
%             % 因为那是 MVALSE 估计的噪声方差, 而不是跟踪器期望的测量方差)
%             tracker_input(2, :) = parameters.measurementVariance;
%             %tracker_input(2, :) = mvalse_output(2,:);
%             % 将转换后的数据存入
%             clutteredMeasurements{step, sensor} = tracker_input;
% 
%         else
%             % MVALSE 没有找到路径，所以该(step, sensor)是空的
%             clutteredMeasurements{step, sensor} = [];
%         end
%     end
% end
% 
% disp('数据转换完成。');
% 
% % (现在，clutteredMeasurements 变量已准备好，
% %  可以传递给您的跟踪算法了)
% Delta_f = 9.980849e+06 ;
Delta_f = 9.965112e+06 ;
N = 512;
min_std = 0.05; 
variance_floor = min_std^2; % 0.0025 m^2
baseline_nsr = 0; 
is_baseline_set = false;
last_mean_variance = zeros(numSensors, 1) + variance_floor;
alpha_smooth = 0.2;

% 设定基础方差 (5cm)
min_std = 0.05;
variance_floor = min_std^2; % 0.0025
for step = 1:numSteps
    for sensor = 1:numSensors
        mvalse_data = estimated_measurements_cell{step, sensor};
        
      if ~isempty(mvalse_data)
            K_est = size(mvalse_data, 2);
            tracker_input = zeros(2, K_est);
            
            % 1. 距离 (m)
            tracker_input(1, :) = mvalse_data(1, :) * c;
            
            % 2. 提取统计量
            nu = mvalse_data(2, :);
            amps = mvalse_data(3, :);
            
            % 3. 计算当前的 NSR
            safe_amps = max(amps, 1e-6); 
            current_nsr = nu ./ (safe_amps.^2);
            
            % 4. 设定基准 NSR (取前20步的平均)
            if ~is_baseline_set && step < 20
                temp_avg = mean(current_nsr);
                if temp_avg > 0
                    baseline_nsr = temp_avg; 
                    is_baseline_set = true;
                    fprintf('--- 基准 NSR 已设定为: %.4e ---\n', baseline_nsr);
                end
            end
            
            % 5. 计算原始自适应方差 (相对比例模型)
            if is_baseline_set && baseline_nsr > 0
                ratio = current_nsr / baseline_nsr;
                ratio = max(ratio, 1.0); % 至少为 1.0
                adaptive_variance = variance_floor * ratio;
            else
                adaptive_variance = variance_floor * ones(1, K_est);
            end
            
            % 6. 应用硬上限 (Clamp Ceiling)
            % 限制最大方差为 0.5m，防止粒子飞得太远
            current_vector = min(adaptive_variance, 0.15);
            
            % 7. [修复后的软着陆] 记忆衰减 (Memory Decay)
            % 逻辑：当前方差不能低于"上一时刻平均方差"的 90%。
            % 这保证了当信号突然变好(600步)时，方差是慢慢降下去的，而不是瞬间跳水。
            
            decay_rate = 0.6; % 衰减率 (值越大，下降越慢)
            prev_level = last_mean_variance(sensor); % 取出上一时刻的标量
            
            % 核心修复：max(向量, 标量) 是合法的 MATLAB 运算
            smoothed_variance = max(current_vector, prev_level * decay_rate);
            
            % 8. 更新历史记录 (保存为标量，供下一步使用)
            last_mean_variance(sensor) = mean(smoothed_variance);
            
            % 9. 赋值
            tracker_input(2, :) = smoothed_variance;
            clutteredMeasurements{step, sensor} = tracker_input;
            
            % (调试输出)
            if mod(step, 50) == 0 && sensor == 1
                fprintf('Step %d: Mean Var = %.4f (Prev=%.4f)\n', ...
                    step, mean(smoothed_variance), prev_level);
            end
      else
            clutteredMeasurements{step, sensor} = zeros(2, 0);

        end
    end
end


% ... 您的后续跟踪代码 (例如 PHD 滤波器) ...
% perform estimation with data association uncertainty
[ estimatedTrajectory, estimatedAnchors, posteriorParticlesAnchors, numEstimatedAnchors ] = BPbasedMINTSLAMnew( dataVA, clutteredMeasurements, parameters, trueTrajectory );

% plot results
plotAll ( trueTrajectory, estimatedTrajectory, estimatedAnchors, posteriorParticlesAnchors{end}, numEstimatedAnchors, dataVA, parameters, 0, parameters.maxSteps );
