# func RenderConversationContextPrompt(messages []event.MessageEvent) (string, string, error) {
# 	builder := strings.Builder{}
# 	builder.WriteString("<conversation_context>\n")

# 	var current string
# 	var latest string
# 	for i := len(messages) - 1; i >= 0; i-- {
# 		msg := messages[i]
# 		content, err := msg.Content()
# 		if err != nil {
# 			return "", "", err
# 		}

# 		contentText, ok := content.(*model.MessageContentText)
# 		if !ok {
# 			return "", "", errx.Internalf("failed to cast MessageContent to MessageContentText")
# 		}

# 		latest = contentText.Text

# 		switch msg.Role {
# 		case model.MessageRoleAgent:
# 			builder.WriteString(fmt.Sprintf("AssistantMessage(%s)\n", contentText.Text))
# 		case model.MessageRoleUser:
# 			builder.WriteString(fmt.Sprintf("AdminMessage(%s)\n", contentText.Text))
# 		case model.MessageRoleCustomer:
# 			builder.WriteString(fmt.Sprintf("UserMessage(%s)\n", contentText.Text))
# 			current = contentText.Text
# 		}
# 	}

# 	if current == "" {
# 		current = latest
# 	}

# 	builder.WriteString("</conversation_context>\n")
# 	builder.WriteString("<current_message_to_analyze>\n")
# 	builder.WriteString(fmt.Sprintf("UserMessage(%s)\n", current))
# 	builder.WriteString("</current_message_to_analyze>\n")
# 	return builder.String(), current, nil
# }