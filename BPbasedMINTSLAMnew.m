% Florian Meyer, Erik Leitinger, 20/05/17.

function [ estimatedTrajectory, estimatedAnchors, posteriorParticlesAnchorsstorage, numEstimatedAnchors ] =  BPbasedMINTSLAMnew( dataVA, clutteredMeasurements, parameters, trueTrajectory )
[numSteps,numSensors] = size(clutteredMeasurements);
numSteps = min(numSteps,parameters.maxSteps);
numParticles = parameters.numParticles;
detectionProbability = parameters.detectionProbability;
priorMean = parameters.priorMean;
survivalProbability = parameters.survivalProbability;
undetectedAnchorsIntensity = parameters.undetectedAnchorsIntensity*ones(numSensors,1);
birthIntensity = parameters.birthIntensity;
clutterIntensity = parameters.clutterIntensity;
unreliabilityThreshold = parameters.unreliabilityThreshold;
execTimePerStep = zeros(numSteps,1);
known_track = parameters.known_track;

load scen_semroom_new;

% allocate memory
estimatedTrajectory = zeros(4,numSteps);
numEstimatedAnchors = zeros(2,numSteps);
storing_idx = 30:30:numSteps;
posteriorParticlesAnchorsstorage = cell(1,length(storing_idx));
% initial state vectors
if(known_track)
  posteriorParticlesAgent = repmat([trueTrajectory(:,1);0;0],1,numParticles);
else
  posteriorParticlesAgent(1:2,:) = drawSamplesUniformlyCirc(priorMean(1:2), parameters.UniformRadius_pos ,parameters.numParticles);
  posteriorParticlesAgent(3:4,:) = repmat(priorMean(3:4),1,parameters.numParticles) + 2*parameters.UniformRadius_vel * rand( 2, parameters.numParticles ) - parameters.UniformRadius_vel;  
end
estimatedTrajectory(:,1) = mean(posteriorParticlesAgent,2);


[ estimatedAnchors, posteriorParticlesAnchors ] =  initAnchors( parameters, dataVA, numSteps, numSensors );
for sensor = 1:numSensors
  numEstimatedAnchors(sensor, 1) = size(estimatedAnchors{sensor,1},2);
end

history_messagesNew=cell(numSensors, numSteps);
history_inputBP = cell(numSensors, numSteps); 
history_newInputBP = cell(numSensors, numSteps);
history_messagelegacy = cell(numSensors, numSteps);
history_weightsAnchor = cell(numSensors, numSteps);

% 在 main loop 之前
if exist('Fnebp_predictions.mat', 'file')
    load('Fnebp_predictions.mat'); % 加载变量 nebp_corrections
    use_nebp = true;
    fprintf('NEBP 增强已开启 \n');
else
    use_nebp = false;
    fprintf('未找到 NEBP 数据，使用原始 BP \n');
end
%% main loop
for step = 2:numSteps
  tic
  % perfrom prediction step
  if(known_track)
    predictedParticlesAgent = repmat([trueTrajectory(:,step);0;0],1,numParticles);
  else
    predictedParticlesAgent = performPrediction( posteriorParticlesAgent, parameters );
  end
  weightsSensors = nan(numParticles,numSensors);
  for sensor = 1:numSensors
    estimatedAnchors{sensor,step} = estimatedAnchors{sensor,step-1};
    measurements = clutteredMeasurements{step,sensor};
    numMeasurements = size(measurements,2);
    
    % predict undetected anchors intensity
    undetectedAnchorsIntensity(sensor) = undetectedAnchorsIntensity(sensor) * survivalProbability + birthIntensity;
    
    % predict "legacy" anchors
    [predictedParticlesAnchors, weightsAnchor] = predictAnchors( posteriorParticlesAnchors{sensor}, parameters );
    
    % create new anchors (one for each measurement)
    [newParticlesAnchors,newInputBP] = generateNewAnchors(measurements, undetectedAnchorsIntensity(sensor) , predictedParticlesAgent, parameters);
    
    % predict measurements from anchors
    [predictedMeasurements, predictedUncertainties, predictedRange] = predictMeasurements(predictedParticlesAgent, predictedParticlesAnchors, weightsAnchor);
    
    % 获取当前步的修正矩阵
    nebp_matrix = [];
    if use_nebp && step <= length(nebp_corrections)
    nebp_matrix = nebp_corrections{step}; 
    end
    % compute association probabilities
    [associationProbabilities, associationProbabilitiesNew, messagelhfRatios, messagesNew,inputbp] = calculateAssociationProbabilitiesGA(measurements, predictedMeasurements, predictedUncertainties, weightsAnchor, newInputBP, parameters,nebp_matrix);
  
    history_inputBP{sensor, step} = inputbp;          % 记录 inputBP
    history_weightsAnchor{sensor,step} = sum(weightsAnchor,1);   % 记录 weightsAnchor
    history_newInputBP{sensor, step} = newInputBP;         % 记录 newInputBP
    history_messagelegacy{sensor, step} = messagelhfRatios;   % 记录 mu_ba (legacy)
    history_messagesNew{sensor, step} = messagesNew;


    % perform message multiplication for agent state
    numAnchors = size(predictedParticlesAnchors,3);
    weights = zeros(numParticles,numAnchors);
    for anchor = 1:numAnchors
      weights(:,anchor) = repmat((1-detectionProbability),numParticles,1);
      for measurement = 1:numMeasurements
        measurementVariance = measurements(2,measurement);
        %factor = 1/sqrt(2*pi*measurementVariance)*detectionProbability; % "Williams style"
        factor = 1/sqrt(2*pi*measurementVariance)*detectionProbability/clutterIntensity; % "BP style"
        weights(:,anchor) = weights(:,anchor) + factor*messagelhfRatios(measurement,anchor)*exp(-1/(2*measurementVariance)*(measurements(1,measurement)-predictedRange(:,anchor)).^2);
      end
      
      % compute anchor predicted existance probability
      predictedExistence = sum(weightsAnchor(:,anchor));
      
      %weights(:,anchor) = weightsAnchor(:,anchor).*weights(:,anchor);
      aliveUpdate = sum(predictedExistence*1/numParticles*weights(:,anchor));
      deadUpdate = 1 - predictedExistence;
      posteriorParticlesAnchors{sensor}{anchor}.posteriorExistence = aliveUpdate/(aliveUpdate+deadUpdate);
      
      % compute anchor belief
      idx_resampling = resampleSystematic(weights(:,anchor)/sum(weights(:,anchor)),numParticles);
      posteriorParticlesAnchors{sensor}{anchor}.x = predictedParticlesAnchors(:,idx_resampling(1:numParticles),anchor);
      posteriorParticlesAnchors{sensor}{anchor}.w = posteriorParticlesAnchors{sensor}{anchor}.posteriorExistence/numParticles*ones(numParticles,1);
      
      estimatedAnchors{sensor,step}{anchor}.x = mean(posteriorParticlesAnchors{sensor}{anchor}.x,2);
      estimatedAnchors{sensor,step}{anchor}.posteriorExistence = posteriorParticlesAnchors{sensor}{anchor}.posteriorExistence;
      
      weights(:,anchor) = predictedExistence*weights(:,anchor) + deadUpdate;
      weights(:,anchor) = log(weights(:,anchor));
      weights(:,anchor) = weights(:,anchor) - max(weights(:,anchor));      
    end
    numEstimatedAnchors(sensor, step) = size(estimatedAnchors{sensor,step},2);
    weightsSensors(:,sensor) = sum(weights,2);
    weightsSensors(:,sensor) = weightsSensors(:,sensor) - max(weightsSensors(:,sensor));
  
    % update undetected anchors intensity
    undetectedAnchorsIntensity(sensor) = undetectedAnchorsIntensity(sensor) * (1-parameters.detectionProbability);
    
    % update new anchors
    for measurement = 1:numMeasurements
      posteriorParticlesAnchors{sensor}{numAnchors+measurement}.posteriorExistence = messagesNew(measurement)*newParticlesAnchors(measurement).constant/(messagesNew(measurement)*newParticlesAnchors(measurement).constant + 1);
      posteriorParticlesAnchors{sensor}{numAnchors+measurement}.x = newParticlesAnchors(measurement).x;
      posteriorParticlesAnchors{sensor}{numAnchors+measurement}.w = posteriorParticlesAnchors{sensor}{numAnchors+measurement}.posteriorExistence/numParticles;
      estimatedAnchors{sensor,step}{numAnchors+measurement}.x = mean(newParticlesAnchors(measurement).x,2);
      estimatedAnchors{sensor,step}{numAnchors+measurement}.posteriorExistence = posteriorParticlesAnchors{sensor}{numAnchors+measurement}.posteriorExistence;
      estimatedAnchors{sensor,step}{numAnchors+measurement}.generatedAt = step;
    end
    
    % delete unreliable anchors
    [estimatedAnchors{sensor,step}, posteriorParticlesAnchors{sensor}] = deleteUnreliableVA( estimatedAnchors{sensor,step}, posteriorParticlesAnchors{sensor}, unreliabilityThreshold );
    numEstimatedAnchors(sensor, step) = size(estimatedAnchors{sensor,step},2);
    history_estimatedAnchors{step} = estimatedAnchors;
  end
  weightsSensors = sum(weightsSensors,2);
  weightsSensors = weightsSensors - max(weightsSensors);
  weightsSensors = exp(weightsSensors);
  weightsSensors = weightsSensors/sum(weightsSensors);
  if(any(storing_idx == step))
    posteriorParticlesAnchorsstorage{storing_idx == step} = posteriorParticlesAnchors;
  end
  if(known_track)
    estimatedTrajectory(:,step) = mean(predictedParticlesAgent,2);
    posteriorParticlesAgent = predictedParticlesAgent;
  else
    estimatedTrajectory(:,step) = predictedParticlesAgent*weightsSensors;
    posteriorParticlesAgent = predictedParticlesAgent(:,resampleSystematic(weightsSensors,numParticles));
  end
  
  % error output
  execTimePerStep(step) = toc;
  error_agent = calcDistance_(trueTrajectory(1:2,step),estimatedTrajectory(1:2,step));
  fprintf('Time instance: %d \n',step);
  fprintf('Number of Anchors Sensor 1: %d \n',numEstimatedAnchors(1, step));
  fprintf('Number of Anchors Sensor 2: %d \n',numEstimatedAnchors(2, step));
  fprintf('Position error agent: %d \n',error_agent);
  fprintf('Execution Time: %4.4f \n',execTimePerStep(step));
  fprintf('--------------------------------------------------- \n\n')
  
% =========================================================================
    % --- [开始] 实时粒子可视化模块 (Real-time Visualization) ---
    % =========================================================================
    
    % 设置绘图频率：每 5 步画一次，避免拖慢运行速度
    % 如果想看每一帧的细节，可以改为 mod(step, 1) == 0
    if mod(step, 5) == 0
        
        % 1. 初始化图窗 (保持句柄，防止闪烁)
        if step == 5 || ~ishandle(100) 
            figure(100); 
            clf; 
            % 设置全屏或大窗口以便观察
            set(gcf, 'Name', 'Particle Evolution Monitor', 'NumberTitle', 'off');
        end
        
        figure(100); 
        clf; hold on; grid on; axis equal;
        
        % ---------------------------------------------------------
        % A. 绘制真实轨迹 (Ground Truth)
        % ---------------------------------------------------------
        % 绘制完整的真实轨迹 (灰色虚线)
        plot(trueTrajectory(1, :), trueTrajectory(2, :), 'Color', [0.7 0.7 0.7], 'LineStyle', '--', 'LineWidth', 1);
        % 绘制当前时刻的真实位置 (黑色十字)
        h_true = plot(trueTrajectory(1, step), trueTrajectory(2, step), 'kx', 'MarkerSize', 12, 'LineWidth', 2);
        
        % ---------------------------------------------------------
        % B. 绘制 Agent 粒子 (Agent Particles)
        % ---------------------------------------------------------
        % 变量名校准: posteriorParticlesAgent (4 x numParticles)
        % 前两行是位置 x, y
        if exist('posteriorParticlesAgent', 'var')
            p_x = posteriorParticlesAgent(1, :);
            p_y = posteriorParticlesAgent(2, :);
            
            % 绘制绿色粒子云
            h_agent_part = plot(p_x, p_y, 'g.', 'MarkerSize', 1);
            
            % 绘制估计均值 (绿色圆圈) - 使用 estimatedTrajectory 存储的值
            est_mean = estimatedTrajectory(1:2, step);
            h_agent_est = plot(est_mean(1), est_mean(2), 'go', 'LineWidth', 2, 'MarkerSize', 8);
        end
        
        % ---------------------------------------------------------
        % C. 绘制 Anchors 粒子 (Legacy & New)
        % ---------------------------------------------------------
        % 变量名校准: posteriorParticlesAnchors (Cell array: {sensor})
        % 内部结构: posteriorParticlesAnchors{sensor}{anchor_idx}
        
        colors = {'r', 'b'}; % 传感器 1 用红色，传感器 2 用蓝色
        
        if exist('posteriorParticlesAnchors', 'var')
            for s = 1:numSensors
                if ~isempty(posteriorParticlesAnchors{s})
                    % 获取该传感器下的所有锚点 (Cell array)
                    sensor_anchors = posteriorParticlesAnchors{s};
                    num_anchors_current = length(sensor_anchors);
                    
                    for k = 1:num_anchors_current
                        % 获取单个锚点结构体
                        anchor_struct = sensor_anchors{k};
                        
                        % 检查结构体有效性
                        if ~isempty(anchor_struct) && isfield(anchor_struct, 'x') && isfield(anchor_struct, 'posteriorExistence')
                            
                            % [关键] 仅绘制存在概率高的特征
                            % 使用 posteriorExistence 字段
                            prob_exist = anchor_struct.posteriorExistence;
                            
                            % 阈值设为 0.5，过滤掉存在概率低的杂波
                            if prob_exist > 0.5
                                % 提取粒子位置 (2 x numParticles)
                                fp_x = anchor_struct.x(1, :);
                                fp_y = anchor_struct.x(2, :);
                                
                                % 绘制特征粒子 (点)
                                plot(fp_x, fp_y, '.', 'Color', colors{mod(s-1,2)+1}, 'MarkerSize', 1);
                                
                                % (可选) 绘制锚点均值 (十字)
                                mean_anchor = mean(anchor_struct.x, 2);
                                plot(mean_anchor(1), mean_anchor(2), '+', 'Color', 'k', 'MarkerSize', 6, 'LineWidth', 1.5);
                            end
                        end
                    end
                end
            end
        end
        
        % ---------------------------------------------------------
        % D. 视图设置
        % ---------------------------------------------------------
        title(sprintf('Time Step: %d / %d', step, numSteps), 'FontSize', 12);
        xlabel('X [m]'); ylabel('Y [m]');
        
        % 固定地图范围 (根据你的 dataVA 调整)
        axis([-10 20 -10 20]); 
        
        legend([h_true, h_agent_est], {'True Agent', 'Est Agent'}, 'Location', 'best');
        
        % [强制刷新] 这一步是让动画动起来的关键
        drawnow limitrate; 
    end
    
    % =========================================================================
    % --- [结束] 实时粒子可视化模块 ---
    % =========================================================================
  
end

% 1. 保存 inputBP
save('data_inputBP.mat', 'history_inputBP');

% 3. 保存 newInputBP
save('data_newInputBP.mat', 'history_newInputBP');

% 4. 保存 messagelegacy (对应 mu_ba)
save('data_messagelegacy.mat', 'history_messagelegacy');

% 5. 保存 messagesNew
save('data_messagesNew.mat', 'history_messagesNew');
save('data_estimatedAnchors.mat','estimatedAnchors')
save('data_weightsAnchor.mat', 'history_weightsAnchor');
end

