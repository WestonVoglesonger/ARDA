# ARDA Documentation

**ARDA (Automated RTL Design Agents)** is an AI-powered system that converts Python algorithms into production-quality SystemVerilog RTL for FPGA implementation.

## 📚 Documentation Overview

This documentation is organized by audience and purpose:

### 👥 User Documentation
- **[User Guide](user_guide.md)** - Getting started, installation, and basic usage
- **[API Reference](api_docs.md)** - Complete API documentation
- **[Examples](examples.md)** - Code examples and tutorials
- **[Troubleshooting](troubleshooting/)** - Common issues and solutions

### 👨‍💻 Developer Documentation
- **[Developer Guide](developer_guide.md)** - Architecture, contributing, and development setup
- **[Architecture](architecture/)** - System architecture and design decisions
- **[Changelog](changelog.md)** - Version history and release notes
- **[Releases](releases/)** - Detailed release notes and phase summaries

### 🔧 Project Documentation
- **[Troubleshooting](troubleshooting/)** - Bug fixes, error tracking, and known issues
- **[Development Notes](development/)** - Implementation details and debugging logs
- **[Architecture Decisions](adr/)** - Architectural decision records
- **[Reviews](reviews/)** - RTL quality reviews by phase

## 🚀 Quick Start

1. **Installation**: See [User Guide](user_guide.md#installation)
2. **First Pipeline Run**: See [Examples](examples.md)
3. **API Usage**: See [API Reference](api_docs.md)

## 📖 Key Topics

### Pipeline Architecture
- **Stages**: Spec → Quant → MicroArch → Architecture → RTL → Verification → Synth → Evaluate
- **Agents**: Specialized AI agents for each pipeline stage
- **Feedback Loop**: Confidence-based iterative improvement

### Key Features
- **Multi-stage AI Pipeline**: Intelligent architectural decisions at each stage
- **Production-Quality RTL**: Synthesis-verified SystemVerilog output
- **Comprehensive Verification**: Functional verification with test vectors
- **Resource Optimization**: FPGA resource budget management

### Development
- **OpenAI Integration**: Uses GPT models with specialized tools
- **Modular Architecture**: Clean separation of concerns
- **Extensible Framework**: Easy to add new algorithms and stages

## 🤝 Contributing

See [Developer Guide](developer_guide.md) for:
- Development setup
- Code organization
- Testing guidelines
- Contribution workflow

## 📋 Recent Changes

See [Changelog](changelog.md) for the latest updates and version history.

## 🔍 Troubleshooting

Having issues? Check:
- [Critical Fixes](troubleshooting/critical-fixes.md) - Known bugs and fixes
- [Error Tracking](troubleshooting/error-tracking.md) - Ongoing issues
- [Verification Issues](troubleshooting/verification-issues.md) - Verification system problems

## 📂 Directory Structure

```
docs/
├── README.md                    # This file - documentation index
├── user_guide.md               # User-facing guide
├── api_docs.md                 # API reference
├── developer_guide.md          # Developer documentation
├── examples.md                 # Code examples
├── troubleshooting.md          # General troubleshooting
├── changelog.md                # Version history
├── architecture.md             # Main architecture overview
├── adr/                        # Architecture Decision Records
├── architecture/               # Detailed architecture docs
├── releases/                   # Release notes and summaries
├── reviews/                    # RTL review results by phase
├── troubleshooting/            # Specific troubleshooting docs
├── development/               # Implementation and debugging notes
└── migrations/                 # Migration and refactoring docs
```

---

**Last Updated:** October 13, 2025
