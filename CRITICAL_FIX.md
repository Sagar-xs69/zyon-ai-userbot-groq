        return [{'role': 'assistant' if r[0]=='assistant' else 'user', 'parts':[{'text':r[1]}]} for r in reversed(rows)]
