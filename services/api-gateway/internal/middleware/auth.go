package middleware

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
)

// PlatformClaims defines the standard enterprise JWT claims structure
type PlatformClaims struct {
	UserID   string   `json:"user_id"`
	Email    string   `json:"email"`
	Roles    []string `json:"roles"`
	TenantID string   `json:"tenant_id"`
	jwt.RegisteredClaims
}

// GenerateToken issues a signed HMAC-SHA256 JWT token for testing and authentication
func GenerateToken(secret string, userID string, email string, roles []string, tenantID string, duration time.Duration) (string, error) {
	now := time.Now()
	claims := PlatformClaims{
		UserID:   userID,
		Email:    email,
		Roles:    roles,
		TenantID: tenantID,
		RegisteredClaims: jwt.RegisteredClaims{
			Subject:   userID,
			Issuer:    "enterprise-ai-platform",
			IssuedAt:  jwt.NewNumericDate(now),
			ExpiresAt: jwt.NewNumericDate(now.Add(duration)),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(secret))
}

// JWTAuthMiddleware validates incoming Bearer tokens and extracts RBAC claims
func JWTAuthMiddleware(secret string, enabled bool, skipPrefixes []string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !enabled {
			c.Next()
			return
		}

		path := c.Request.URL.Path
		for _, prefix := range skipPrefixes {
			if strings.HasPrefix(path, prefix) {
				c.Next()
				return
			}
		}

		authHeader := c.GetHeader("Authorization")
		traceID := c.GetString("trace_id")

		if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error":    "Unauthorized",
				"message":  "Authorization header with Bearer token is required",
				"trace_id": traceID,
			})
			return
		}

		tokenString := strings.TrimPrefix(authHeader, "Bearer ")

		token, err := jwt.ParseWithClaims(tokenString, &PlatformClaims{}, func(token *jwt.Token) (interface{}, error) {
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
			}
			return []byte(secret), nil
		})

		if err != nil || !token.Valid {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error":    "Unauthorized",
				"message":  "Invalid or expired JWT token",
				"details":  err.Error(),
				"trace_id": traceID,
			})
			return
		}

		claims, ok := token.Claims.(*PlatformClaims)
		if !ok {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error":    "Unauthorized",
				"message":  "Malformed token claims",
				"trace_id": traceID,
			})
			return
		}

		// Inject user and tenant context into Gin context
		c.Set("user_id", claims.UserID)
		c.Set("email", claims.Email)
		c.Set("roles", claims.Roles)
		c.Set("tenant_id", claims.TenantID)

		// Inject downstream headers so downstream services receive user context securely
		c.Request.Header.Set("X-User-Id", claims.UserID)
		c.Request.Header.Set("X-Tenant-Id", claims.TenantID)
		c.Request.Header.Set("X-User-Roles", strings.Join(claims.Roles, ","))

		c.Next()
	}
}

// RequireRole enforces Role-Based Access Control (RBAC) on routes
func RequireRole(allowedRoles ...string) gin.HandlerFunc {
	return func(c *gin.Context) {
		rawRoles, exists := c.Get("roles")
		if !exists {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"error":    "Forbidden",
				"message":  "No user roles found in security context",
				"trace_id": c.GetString("trace_id"),
			})
			return
		}

		userRoles, ok := rawRoles.([]string)
		if !ok {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"error":    "Forbidden",
				"message":  "Invalid user roles type in security context",
				"trace_id": c.GetString("trace_id"),
			})
			return
		}

		for _, required := range allowedRoles {
			for _, role := range userRoles {
				if role == required || role == "admin" {
					c.Next()
					return
				}
			}
		}

		c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
			"error":          "Forbidden",
			"message":        "Insufficient permissions to access this endpoint",
			"required_roles": allowedRoles,
			"user_roles":     userRoles,
			"trace_id":       c.GetString("trace_id"),
		})
	}
}
