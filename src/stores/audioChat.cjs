import { ref } from 'vue'
import { defineStore } from 'pinia'

export const useAudioChatStore = defineStore('audioChat', () => {
  const file = ref({})
  const gptResponse = ref('')
  const isLoadingGPT = ref(false)
  const prompt = ref([])
  const question = ref('')
  const questionAnswerList = ref([])
  const transcript = ref('')

  function transcribeFile() {
    const formData = new FormData()
    formData.append('file', file.value.value)

    fetch('http://localhost:3000/dg-transcription', {
      method: 'POST',
      body: formData
    })
      .then((response) => response.json())
      .then((data) => {
        transcript.value = data.transcript.results.channels[0].alternatives[0].transcript
      })
  }

  function createPrompt() {
    const instructions = {
      role: 'system',
      content:
        'You will answer questions about the following text that has been transcribed from an audio file.'
    }
    const transcriptToAnalyze = { role: 'user', content: transcript.value }
    const chatQuestion = { role: 'user', content: question.value }

    ///create prompt array
    prompt.value.push(instructions)
    prompt.value.push(transcriptToAnalyze)
    prompt.value.push(chatQuestion)

    sendPrompt()
  }

  function sendPrompt() {
    fetch('http://localhost:3000/chat', {
      method: 'POST',
      body: JSON.stringify({
        messages: prompt.value
      }),
      headers: {
        'Content-Type': 'application/json'
      }
    })
      .then((response) => response.json())
      .then((data) => {
        isLoadingGPT.value = false
        gptResponse.value = data.message.content
        // array to save the conversation
        questionAnswerList.value.push({
          question: question.value,
          answer: data.message.content
        })
        question.value = ''
      })
  }

  return { file, transcript, prompt, question, questionAnswerList, transcribeFile, createPrompt }
})
