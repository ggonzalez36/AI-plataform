package config

import (
	"os"
	"strconv"
)

// Config holds all runtime configuration for the API Gateway
type Config struct {
	Port            string
	RagServiceURL   string
	MLOpsServiceURL string
	RedisAddr       string
	RedisPassword   string
	JWTSecret       string
	RateLimitRPS    int
	RateLimitBurst  int
	AuthEnabled     bool
	Environment     string
}

// LoadConfig reads configuration from environment variables with production-grade defaults
func LoadConfig() *Config {
	return &Config{
		Port:            getEnv("PORT", "8080"),
		RagServiceURL:   getEnv("RAG_SERVICE_URL", "http://hybrid-rag-engine:8001"),
		MLOpsServiceURL: getEnv("MLOPS_SERVICE_URL", "http://mlops-inference:8002"),
		RedisAddr:       getEnv("REDIS_ADDR", "redis:6379"),
		RedisPassword:   getEnv("REDIS_PASSWORD", ""),
		JWTSecret:       getEnv("JWT_SECRET", "enterprise-platform-secret-key-2026-min-32-chars-long"),
		RateLimitRPS:    getEnvInt("RATE_LIMIT_RPS", 50),
		RateLimitBurst:  getEnvInt("RATE_LIMIT_BURST", 100),
		AuthEnabled:     getEnvBool("AUTH_ENABLED", true),
		Environment:     getEnv("ENVIRONMENT", "development"),
	}
}

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

func getEnvInt(key string, defaultVal int) int {
	if val := os.Getenv(key); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil {
			return parsed
		}
	}
	return defaultVal
}

func getEnvBool(key string, defaultVal bool) bool {
	if val := os.Getenv(key); val != "" {
		if parsed, err := strconv.ParseBool(val); err == nil {
			return parsed
		}
	}
	return defaultVal
}
