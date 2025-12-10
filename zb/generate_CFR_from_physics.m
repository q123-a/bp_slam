function [Y_noisy_CFR_cell, H_true_CFR_cell, true_delays_cell, f_subcarrier_new] = generate_CFR_from_physics(targetTrajectory, dataVA, parameters, model_params)
%
% 描述: (已更新)
%   1. 扫描所有传感器和 *所有时间步* 以找到全局最大时延 tau_max。
%   2. 根据 tau_max 设定一个安全的子载波间隔 f_new = 1 / (2 * tau_max)。
%   3. 使用 f_new 构建所有 (step, sensor) 组合的 CFR。
%
% 输出: (已更新)
%   Y_noisy_CFR_cell{step, sensor}: 嘈杂的 CFR
%   H_true_CFR_cell{step, sensor}:  真实的 CFR
%   true_delays_cell{step, sensor}: 真实的物理时延
%   f_subcarrier_new:    新计算的、避免混叠的子载波间隔 (Hz)
%

% --- 1. 初始化参数 ---
c = 3.0e8; % 光速

% (新) 从 model_params 获取 numSteps
numSteps = model_params.numSteps; 

% 从 MVALSE 模型中获取参数
N_subcarrier = model_params.N_subcarrier;
f_carrier = model_params.f_s;
snr_db = model_params.snr;

m_indices = (0:N_subcarrier-1).';
numSensors = length(dataVA); 

% (新) 将 Cell 更改为 (numSteps x numSensors)
true_delays_cell = cell(numSteps, numSensors);
true_amps_cell   = cell(numSteps, numSensors);
global_max_tau   = 0; % 用于跟踪最大时延

% --- 2. (新) 第一次循环: 嵌套循环 (step, sensor) 查找 tau_max ---
fprintf('--- 物理仿真: 正在扫描 %d 步 x %d 传感器以查找 tau_max ---\n', numSteps, numSensors);

for step = 1:numSteps
    % (新) 在 step 循环内部更新目标位置
    target_pos = targetTrajectory(:, step); 
    
    for sensor = 1:numSensors
        positions = dataVA{sensor}.positions;
        visibility = dataVA{sensor}.visibility;
        [~, numAnchors] = size(positions);

        true_delays_sensor = [];
        true_amps_sensor = [];

        for anchor = 1:numAnchors
            % (新) 确保使用正确的 visibility 索引
            if(visibility(anchor, step)) 
                % (a) 计算真实距离和时延
                anchor_pos = positions(:, anchor);
                dist = sqrt((anchor_pos(1) - target_pos(1)).^2 + (anchor_pos(2) - target_pos(2)).^2);
                tau_k = dist / c; % 真实时延
                
                % (b) 计算增益
                sigma_k_sq = c / (4 * pi * dist * f_carrier); 
                alpha_k = sqrt(sigma_k_sq/2) * (randn() + 1i * randn()); 

                % (c) 存储
                true_delays_sensor = [true_delays_sensor; tau_k];
                true_amps_sensor = [true_amps_sensor; alpha_k];
                
                % (d) 检查全局最大值
                if tau_k > global_max_tau
                    global_max_tau = tau_k;
                end
            end
        end
        
        true_delays_cell{step, sensor} = true_delays_sensor;
        true_amps_cell{step, sensor} = true_amps_sensor;
    end
end

% --- 3. 根据 tau_max 设定 f ---
fprintf('--- 物理仿真: 检测到全局最大时延 tau_max = %e s ---\n', global_max_tau);
f_subcarrier_new = 1 / (2 * global_max_tau);
fprintf('--- 信号处理: 已设定子载波间隔 f = %e Hz 以避免混叠 ---\n', f_subcarrier_new);

% --- 4. (新) 第二次循环: 嵌套循环 (step, sensor) 构建 CFR ---
Y_noisy_CFR_cell = cell(numSteps, numSensors);
H_true_CFR_cell  = cell(numSteps, numSensors);

for step = 1:numSteps
    for sensor = 1:numSensors
    
        % (新) 从 cell{step, sensor} 中检索
        true_delays_sensor = true_delays_cell{step, sensor};
        true_amps_sensor = true_amps_cell{step, sensor};
        K_paths = length(true_delays_sensor);
        
        H_true_CFR_sensor = zeros(N_subcarrier, 1);

        for k = 1:K_paths
            tau_k = true_delays_sensor(k);
            alpha_k = true_amps_sensor(k);
            
            % 使用新计算的 f_subcarrier_new
            s_tau_k = 2 * pi * f_subcarrier_new * tau_k; 
            H_true_CFR_sensor = H_true_CFR_sensor + alpha_k * exp(-1i * m_indices * s_tau_k);
        end

        % 添加噪声
        Pn = mean(abs(H_true_CFR_sensor).^2) * 10^(-snr_db/10);
        eps = sqrt(0.5 * Pn) * (randn(N_subcarrier,1) + 1i * randn(N_subcarrier,1));
        Y_noisy_CFR_sensor = H_true_CFR_sensor + eps;

        % (新) 存储到 cell{step, sensor}
        Y_noisy_CFR_cell{step, sensor} = Y_noisy_CFR_sensor;
        H_true_CFR_cell{step, sensor}  = H_true_CFR_sensor;
        
    end
end % (传感器循环结束)

end