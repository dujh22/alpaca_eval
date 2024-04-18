# 客户端配置

客户端配置是 json yaml，可将模型名称映射到用于实例化 OpenAI 等客户端的配置列表。
我们使用列表允许在达到速率限制时切换客户端（例如，使用不同的组织 ID 或使用 Azure）。

## 配置 OpenAI

要使用新的 OpenAI 配置，请在当前目录下创建 `openai_configs.yaml`。配置应该是一个字典，其值是 OpenAI 客户端的配置列表。我们使用该列表允许在达到速率限制时切换 OpenAI 客户端（例如，使用不同的组织 ID 或使用 Azure）。

如果不需要在 OpenAI 客户端之间切换，下面是最简单的配置：

```yaml
default:
    - api_key: "<your OpenAI API key here>"
      organization: "<your organization ID>"
```

如果您想在遇到费率限制时在不同组织 ID 之间切换，请使用下面的方法：

```yaml
default:
    - api_key: "<your OpenAI API key here>"
      organization: "<your 1st organization ID>"

    - api_key: "<your OpenAI API key here>"
      organization: "<your 2nd organization ID>"
```

请注意，顺序并不重要：我们将随机选择客户端。这样可以并行运行多个作业，同时减少使用同一客户端的机会。

有时，您可能需要特定于模型并使用不同客户端类的配置，例如在使用 Azure 客户端时。在这种情况下，可以执行以下操作：

```yaml
default:
    - api_key: "<your OpenAI API key here>"
      organization: "<your 1st organization ID>"

    - api_key: "<your OpenAI API key here>"
      organization: "<your 2nd organization ID>"

gpt-4-1106-preview: # only when using `model_name: gpt-4-1106-preview`
    - "default" # this will append all the `default` configs
  
    - client_class: "openai.AzureOpenAI" # doesn't use the `openai.OpenAI` client class
      # the following are passed to the `AzureOpenAI` client class
      azure_deployment: "gpt-4-1106"
      api_key: "<your Azure OpenAI API key here>"
      azure_endpoint: "<your Azure OpenAI API base here>"
      api_version: "2023-07-01-preview"
```

在这里，当在 evaluators_configs 中使用 model_name gpt-4 时，配置将被附加到默认值，例如[这里](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4/configs.yaml#L6)。当遇到速率限制时，我们将在两个 OpenAI 客户端和一个 Azure 客户端之间切换，每个客户端都使用相同的底层模型。请注意，在使用 Azure 时，某些参数可能会略有不同，从而导致问题，因为 Azure 通常比 OpenAI 的 API 滞后几个月。遗憾的是，由于 Azure 不提供 logprobs，因此目前无法在 AlpacaEval2.0 中使用 Azure。

## 完全向后兼容

在 `alpaca_eval==0.3.7` 之前，设置客户端的推荐方法是使用环境变量 `OPENAI_API_KEYS` / `OPENAI_ORGANIZATION_IDS`，它们是逗号分隔的常量列表。**使用这些变量仍然有效，但会引发警告。**在引擎盖下，如果

1. openai_configs.yaml` 不存在，并且
2. 环境变量已设置

则结果基本上与以下配置一致（键值应展开）：

```yaml
default:
- api_key: "<OPENAI_API_KEYS[0]>"
  organization: "<OPENAI_ORGANIZATION_IDS[0]>"

- api_key: "<OPENAI_API_KEYS[1]>"
  organization: "<OPENAI_ORGANIZATION_IDS[1]>"

...
```
