import dotenv from 'dotenv'
import express from 'express'
import cors from 'cors'
import bodyParser from 'body-parser'
import multer from 'multer'
import OpenAI from 'openai'
import { encode } from 'gpt-3-encoder'
import deepgramPkg from '@deepgram/sdk'
import Replicate from 'replicate'

import { OpenAI as LangchainOpenAI } from 'langchain/llms/openai'
// import { PromptTemplate } from 'langchain/prompts'
import { ConversationChain } from 'langchain/chains'
import { BufferMemory } from 'langchain/memory'

const envConfig = dotenv.config()
const upload = multer()
const { Deepgram } = deepgramPkg
const deepgram = new Deepgram(process.env.DG_API)

const port = 3000
const app = express()
app.use(cors())
app.use(bodyParser.json())

const configuration = {
  apiKey: process.env.OPENAI_API_KEY
}

const openai = new OpenAI(configuration)

const replicate = new Replicate({
  auth: process.env.REPLICATE
})

app.post('/chat', async (req, res) => {
  const messages = req.body.messages
  try {
    if (messages == null) {
      throw new Error('We have a problem - no prompt was provided')
    }

    const response = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages
    })
    const completion = response.choices[0].message

    return res.status(200).json({
      success: true,
      message: completion
    })
  } catch (error) {
    console.log(error.message)
  }
})

app.post('/tokenize', async (req, res) => {
  const str = req.body.stringToTokenize

  try {
    if (str == null) {
      throw new Error('No string was provided')
    }
    const encoded = encode(str)
    const length = encoded.length
    console.log('Token count is ' + length)
    return res.status(200).json({
      success: true,
      tokens: length
    })
  } catch (error) {
    console.log(error.message)
  }
})

app.post('/dg-transcription', upload.single('file'), async (req, res) => {
  try {
    const dgResponse = await deepgram.transcription.preRecorded(
      {
        buffer: req.file.buffer,
        mimetype: req.file.mimetype
      },
      {
        punctuate: true,
        model: 'nova'
      }
    )
    res.send({ transcript: dgResponse })
  } catch (e) {
    console.log('error', e)
  }
})

const miniGPT =
  'daanelson/minigpt-4:b96a2f33cc8e4b0aa23eacfce731b9c41a7d9466d9ed4e167375587b54db9423'

// Replicate (minigpt) image analyzer
app.post('/minigpt', async (req, res) => {
  try {
    const miniGPTResponse = await replicate.run(miniGPT, {
      input: {
        image: req.body.image,
        prompt: req.body.prompt
      }
    })
    res.send({ message: miniGPTResponse })
  } catch (e) {
    console.log('error', e)
  }
})

const model = new LangchainOpenAI({})
const memory = new BufferMemory()
const chain = new ConversationChain({ llm: model, memory: memory })
let chainNum = 0

app.post('/chain', async (req, res) => {
  chainNum++
  const messages = req.body.messages

  if (chainNum === 1) {
    const firstResponse = await chain.call({ input: messages[0].content })
    console.log(firstResponse)
    const secondResponse = await chain.call({ input: messages[1].content })
    console.log(secondResponse)
    const thirdResponse = await chain.call({ input: messages[2].content })
    console.log(thirdResponse)
    return res.status(200).json({
      success: true,
      message: thirdResponse.response
    })
  } else {
    const nextResponse = await chain.call({ input: messages[2].content })
    console.log(nextResponse)
    return res.status(200).json({
      success: true,
      message: nextResponse.response
    })
  }
})

app.get('/clear-chain', async (req, res) => {
  memory.clear()
  chainNum = 0
  return res.status(200).json({
    success: true,
    message: 'Memory is clear!'
  })
})

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
