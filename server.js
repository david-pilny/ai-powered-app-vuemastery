import dotenv from 'dotenv'
import express from 'express'
import cors from 'cors'
import bodyParser from 'body-parser'
import OpenAI from 'openai'
const envConfig = dotenv.config()

const port = 3000
const app = express()
app.use(cors())
app.use(bodyParser.json())

const configuration = {
  apiKey: process.env.OPENAI_API_KEY
}

const openai = new OpenAI(configuration)

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

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
