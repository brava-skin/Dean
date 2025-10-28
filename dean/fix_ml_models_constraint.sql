ALTER TABLE ml_models DROP CONSTRAINT IF EXISTS uk_ml_models_type_stage;
ALTER TABLE ml_models ADD CONSTRAINT uk_ml_models_type_stage_version UNIQUE (model_type, stage, version);
