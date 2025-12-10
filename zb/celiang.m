clear;
clc;

% --- 1. 定义物理和 MVALSE 参数 ---
N_subcarrier = 512;
f_s = 28e9;        % 载波频率
c = 3.0e8;
snr = 10;          % 固定SNR
%numSensors = 2; 

% --- 2. 加载您的 .mat 文件 ---
parameters.known_track = 0;  % 是否已知轨迹（0表示未知轨迹）

load('scenarioCleanM2_new901.mat'); % 加载场景数据，包括虚拟锚点 dataVA 和真实轨迹 trueTrajectory

% 将所有锚点的可见性设置为全可见（1）
[numSensors, ~] = size(dataVA);
for sensor = 1:numSensors
  dataVA{sensor}.visibility = ones(size(dataVA{sensor}.visibility,1), length(trueTrajectory));
end

% ---------------------------
% 2. 算法参数配置
% ---------------------------
parameters.maxSteps = 900;          % 最大时间步数
trueTrajectory = trueTrajectory(:,1:parameters.maxSteps); % 取前maxSteps个时间步的轨迹
parameters.lengthStep = 0.03;       % 单步移动距离（米）
parameters.scanTime = 1;             % 采样时间间隔（秒）

% 最大速度和过程噪声方差计算
v_max = parameters.lengthStep / parameters.scanTime;
parameters.drivingNoiseVariance = (v_max / 3 / parameters.scanTime)^2;

% 测量噪声参数
parameters.measurementVariance = 0.1^2;      % 距离测量方差
parameters.measurementVarianceLHF = 0.15^2;  % 后验测量方差（用于LHF）

% 检测概率
parameters.detectionProbability = 0.95;

% 区域尺寸及杂波相关参数
parameters.regionOfInterestSize = 30;               % 区域边长（米）
parameters.meanNumberOfClutter = 1;                  % 平均误报数
parameters.clutterIntensity = parameters.meanNumberOfClutter / parameters.regionOfInterestSize; % 杂波强度

% 新锚点出生率
parameters.meanNumberOfBirth = 1e-4;
parameters.birthIntensity = parameters.meanNumberOfBirth / (2 * parameters.regionOfInterestSize)^2;

% 未检测锚点强度
parameters.meanNumberOfUndetectedAnchors = 6;
parameters.undetectedAnchorsIntensity = parameters.meanNumberOfUndetectedAnchors / (2 * parameters.regionOfInterestSize)^2;

% 粒子滤波相关参数
parameters.numParticles = 100000;   % 粒子数量
parameters.upSamplingFactor = 1;    % 粒子上采样因子

% SLAM相关阈值与先验
parameters.detectionThreshold = 0.5;
parameters.survivalProbability = 0.999;  % 锚点存活概率
parameters.unreliabilityThreshold = 1e-4; % 锚点存在概率阈值，低于则删除
parameters.priorKnownAnchors{1} = 1;      % 传感器1已知锚点索引
parameters.priorKnownAnchors{2} = 1;      % 传感器2已知锚点索引
parameters.priorCovarianceAnchor = 0.001^2 * eye(2); % 锚点位置先验协方差
parameters.anchorRegularNoiseVariance = 1e-4^2;      % 锚点过程噪声方差

% agent参数（均匀采样半径）
parameters.UniformRadius_pos = 0.5;  % 初始位置均匀采样半径
parameters.UniformRadius_vel = 0.05; % 初始速度均匀采样半径

% ---------------------------
% 3. 随机种子设置（保证结果可重复）
% ---------------------------
rng(1)

% ---------------------------
% 4. 移动体初始位置均值设定（真实轨迹起点）
% ---------------------------
parameters.priorMean = [trueTrajectory(1:2,1); 0; 0]; % 初始位置+速度
if length(dataVA) < numSensors
    error('dataVA 中的传感器数量 %d 少于要求的 %d', length(dataVA), numSensors);
end
dataVA = dataVA(1:numSensors);
targetTrajectory = trueTrajectory;
% (新) 从 targetTrajectory 中读取 numSteps
[~, numSteps] = size(targetTrajectory);
fprintf('--- 检测到 %d 个时间步 ---\n', numSteps);

% 将 MVALSE 参数打包
model_params.N_subcarrier = N_subcarrier;
model_params.f_s = f_s;
model_params.snr = snr;
model_params.numSteps = numSteps; % (新) 传入 numSteps

tic
% 你的代码块
% --- 3. (新) 生成 CFR 和真值 (并接收 f_new) ---
[Y_noisy_CFR_cell, H_true_CFR_cell, true_delays_cell, f_new] = ...
    generate_CFR_from_physics(targetTrajectory, dataVA, parameters, model_params);

% --- 4. (新) 嵌套循环: 遍历所有 (step, sensor) ---

Mcal_freq = (0:N_subcarrier - 1).';

% (新) 结果存储更改为 {step, sensor}
estimated_measurements_cell = cell(numSteps, numSensors);

for step = 1:numSteps
    for sensor = 1:numSensors
        
        %fprintf('\n--- 步骤 %d / %d, 传感器 %d: 正在估计时延 ---\n', step, numSteps, sensor);
        
        % (a) 提取该 (step, sensor) 的数据
        Y_in = Y_noisy_CFR_cell{step, sensor};
        H_in = H_true_CFR_cell{step, sensor};
        tau_true = true_delays_cell{step, sensor};
        
        % (b) 检查：如果这个(step, sensor)没有可见路径，则跳过
        if isempty(tau_true)
            %fprintf('--- 步骤 %d / %d, 传感器 %d: 没有可见路径, 跳过 ---\n', step, numSteps, sensor);
            estimated_measurements_cell{step, sensor} = []; % 存入空值
            continue;
        end

        % (c) 运行 MVALSE
        out_Delay = MVALSE(Y_in, Mcal_freq, 2, H_in);
        
       % fprintf('--- 步骤 %d / %d, 传感器 %d: 时延对比 ---\n', step, numSteps, sensor);
        s_tau_est = out_Delay.freqs; 

        % (d) 反推时延
        tau_est_sec = sort(-s_tau_est / (2 * pi * f_new)); 
        tau_true_sec = sort(tau_true); 
        
        %disp('     真实值 (秒)     |   估计值 (秒)');
        max_len = max(length(tau_true_sec), length(tau_est_sec));
        tau_true_padded = [tau_true_sec; NaN(max_len - length(tau_true_sec), 1)];
        tau_est_padded = [tau_est_sec; NaN(max_len - length(tau_est_sec), 1)];
        disp([tau_true_padded, tau_est_padded]);

        % (e) 存储为您要求的格式
K_est = length(tau_est_sec);
measurements_sensor = zeros(3, K_est); % 变为 3 行

measurements_sensor(1, :) = tau_est_sec.';         % 第1行: 时延 (秒)
measurements_sensor(2, :) = out_Delay.noise_var;   % 第2行: 输入噪声功率 (nu)
measurements_sensor(3, :) = abs(out_Delay.amps).'; % 第3行: 路径幅度的模 (用于计算SNR)

estimated_measurements_cell{step, sensor} = measurements_sensor;

    end
end

elapsedTime = toc;
disp(['运行时间: ', num2str(elapsedTime)]);
% --- 5. 显示最终输出 ---
% fprintf('\n--- 所有传感器的估计测量结果 (Cell 数组) ---\n');
% disp('--- 维度: {numSteps, numSensors} ---');
% disp(estimated_measurements_cell);